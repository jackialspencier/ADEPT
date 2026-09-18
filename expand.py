import json
import os
from collections import OrderedDict
from typing import TYPE_CHECKING

import fire
import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.modeling_utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME

if TYPE_CHECKING:
    from transformers import PretrainedConfig

def change_name(name: str, old_index: int, new_index: int) -> str:
    """Replace the layer index in a weight name from old_index to new_index."""
    return name.replace(f".{old_index:d}.", f".{new_index:d}.")

def _parse_expand_layers(expand_layers) -> list[int]:
    """Parse Fire CLI args: "2,5,8" may arrive as str, or as tuple (2, 5, 8)."""
    if isinstance(expand_layers, (list, tuple)):
        return [int(x) for x in expand_layers]
    if isinstance(expand_layers, int):
        return [expand_layers]
    text = str(expand_layers).strip()
    if not text:
        raise ValueError("expand_layers is empty")
    # Strip accidental quotes/parentheses from shell/Fire
    text = text.strip("()[]\"'")
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def block_expansion(
    model_name_or_path: str,
    output_dir: str,
    expand_layers: str,  # e.g. "2,5,8" (Fire may also pass tuple (2,5,8))
    shard_size: str = "5GB",
    save_safetensors: bool = True,
):
    r"""Perform block expansion for LLaMA, Mistral, Qwen2, Yi, or Gemma models.

    Usage: python expand.py --model_name_or_path meta-llama/Llama-2-7b-hf --output_dir /Your/Path/To/llama2_pro --expand_layers="2,5,8"
    """
    config: PretrainedConfig = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    num_layers = getattr(config, "num_hidden_layers")

    print("expand_layers:", expand_layers)
    expand_layer_list = _parse_expand_layers(expand_layers)
    num_expand = len(expand_layer_list)
 
    # Validate that the specified layers are within valid range
    if any(layer >= num_layers or layer < 0 for layer in expand_layer_list):
        raise ValueError(f"Expand layers must be between 0 and {num_layers-1}")
    if len(set(expand_layer_list)) != len(expand_layer_list):
        raise ValueError("Duplicate layers in expand_layers")
    
    # Update config with new number of layers.
    # Gemma2+/transformers>=5 keep a per-layer `layer_types` list that must match
    # `num_hidden_layers`. Inserted blocks inherit the source layer's attention type.
    old_layer_types = list(getattr(config, "layer_types", None) or [])
    if old_layer_types and len(old_layer_types) != num_layers:
        raise ValueError(
            f"config.layer_types length ({len(old_layer_types)}) != num_hidden_layers ({num_layers})"
        )
    new_layer_types: list[str] = []
    expand_set_for_types = set(expand_layer_list)
    for i in range(num_layers):
        src_type = old_layer_types[i] if old_layer_types else None
        if src_type is None:
            # Gemma2 default: odd positions (1-based) sliding, even full
            src_type = "sliding_attention" if bool((i + 1) % 2) else "full_attention"
        new_layer_types.append(src_type)
        if i in expand_set_for_types:
            new_layer_types.append(src_type)

    setattr(config, "num_hidden_layers", num_layers + num_expand)
    if hasattr(config, "layer_types") or old_layer_types:
        setattr(config, "layer_types", new_layer_types)
    config.save_pretrained(output_dir)

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    print(f"Expanding model of {num_layers} layers to {num_layers + num_expand} layers.")
    print(f"Expanding after layers: {expand_layers}")
    
    # Load original model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype="auto", device_map="cpu", trust_remote_code=True, low_cpu_mem_usage=True
    )
    assert isinstance(model, PreTrainedModel)  # type hint
    if save_safetensors and getattr(model.config, "tie_word_embeddings", False):
        del model.lm_head  # safetensors does not allow shared weights

    state_dict = model.state_dict()
    output_state_dict: dict[str, torch.Tensor] = OrderedDict()
    
    # Create mapping from original layer indices to new layer indices
    layer_mapping = {}
    new_layer_idx = 0
    expand_set = set(expand_layer_list)
    
    # Build layer mapping: for each original layer, assign one or two new indices
    for i in range(num_layers):
        layer_mapping[i] = new_layer_idx
        new_layer_idx += 1
        if i in expand_set:
            # If this layer is to be expanded, insert an additional layer
            layer_mapping[f"{i}_expanded"] = new_layer_idx
            new_layer_idx += 1
    
    # Copy and expand layers according to the mapping
    for i in range(num_layers):
        # Copy original layer weights
        for key, value in state_dict.items():
            if f".{i:d}." in key:
                new_key = change_name(key, i, layer_mapping[i])
                output_state_dict[new_key] = value
        print(f"Add layer {layer_mapping[i]} copied from layer {i}.")
        
        # If expansion is needed for this layer
        if i in expand_set:
            for key, value in state_dict.items():
                if f".{i:d}." in key:
                    new_key = change_name(key, i, layer_mapping[f"{i}_expanded"])
                    # Initialize expansion-specific weights to zero for down_proj and o_proj
                    if "down_proj" in key or "o_proj" in key:
                        output_state_dict[new_key] = torch.zeros_like(value)
                    else:
                        output_state_dict[new_key] = torch.clone(value)
            print(f"Add layer {layer_mapping[f'{i}_expanded']} expanded from layer {i}.")

    # Copy non-layer parameters (e.g., embeddings, norm, lm_head)
    for key, value in state_dict.items():
        if key not in output_state_dict:
            output_state_dict[key] = value

    # Save weights
    weights_name = SAFE_WEIGHTS_NAME if save_safetensors else WEIGHTS_NAME
    filename_pattern = weights_name.replace(".bin", "{suffix}.bin").replace(".safetensors", "{suffix}.safetensors")
    state_dict_split = split_torch_state_dict_into_shards(
        output_state_dict, filename_pattern=filename_pattern, max_shard_size=shard_size
    )
    
    for shard_file, tensors in tqdm(state_dict_split.filename_to_tensors.items(), desc="Save weights"):
        shard = {tensor: output_state_dict[tensor].contiguous() for tensor in tensors}
        if save_safetensors:
            save_file(shard, os.path.join(output_dir, shard_file), metadata={"format": "pt"})
        else:
            torch.save(shard, os.path.join(output_dir, shard_file))

    if not state_dict_split.is_sharded:
        print(f"Model weights saved in {os.path.join(output_dir, weights_name)}.")
    else:
        index = {
            "metadata": state_dict_split.metadata,
            "weight_map": state_dict_split.tensor_to_filename,
        }
        index_name = SAFE_WEIGHTS_INDEX_NAME if save_safetensors else WEIGHTS_INDEX_NAME
        with open(os.path.join(output_dir, index_name), "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, sort_keys=True)

        print(f"Model weights saved in {output_dir}.")

    print("- Fine-tune this model with:")
    print(f"model_name_or_path: {output_dir}")
    print("finetuning_type: freeze")
    print(f"freeze_trainable_layers: {num_expand}")
    print("use_llama_pro: true")

if __name__ == "__main__":
    fire.Fire(block_expansion)