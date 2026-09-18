import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import copy

def _distributed_requested() -> bool:
    """True only when launched via torchrun/accelerate (env rendezvous present)."""
    if dist.is_available() and dist.is_initialized():
        return True
    # torchrun sets LOCAL_RANK and MASTER_ADDR; plain `python` should stay single-process.
    if os.environ.get("MASTER_ADDR") and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        return True
    if os.environ.get("RANK") is not None and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        return True
    return False


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def setup_distributed():
    """Init NCCL only for multi-process launches; plain `python` stays single-GPU."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for calc_importance.py")

    num_gpus = torch.cuda.device_count()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if _distributed_requested():
        if "WORLD_SIZE" not in os.environ:
            os.environ["WORLD_SIZE"] = str(num_gpus)
        if "RANK" not in os.environ:
            os.environ["RANK"] = "0"
        if "LOCAL_RANK" not in os.environ:
            os.environ["LOCAL_RANK"] = "0"
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        return local_rank, dist.get_world_size()

    # Single-process path (recommended for nohup + CUDA_VISIBLE_DEVICES=0)
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    torch.cuda.set_device(local_rank)
    return local_rank, 1

class SimpleDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        if isinstance(item, dict):
            if "text" in item:
                text = item["text"]
                encoded = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt"
                )
                return {
                    "input_ids": encoded["input_ids"][0],
                    "attention_mask": encoded["attention_mask"][0],
                    "labels": encoded["input_ids"][0],
                    "data_type": "pt"
                }
            elif "instruction" in item and "output" in item:
                text = f"User: {item['instruction']}\nAssistant: {item['output']}"
                encoded = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt"
                )
                
                instruction_part = f"User: {item['instruction']}\nAssistant:"
                instruction_tokens = self.tokenizer(
                    instruction_part,
                    add_special_tokens=False
                ).input_ids
                instruction_len = len(instruction_tokens)
                
                labels = encoded["input_ids"][0].clone()
                labels[:instruction_len] = -100
                
                return {
                    "input_ids": encoded["input_ids"][0],
                    "attention_mask": encoded["attention_mask"][0],
                    "labels": labels,
                    "data_type": "sft"
                }

def create_dataloader(dataset, batch_size, local_rank):
    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=False)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
        )
        return dataloader, sampler

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return dataloader, None

def gradient_importance_analysis(model, dataloader, device):
    importance = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            importance[name] = 0.0

    model.train()
    sample_counts = {"pt": 0, "sft": 0}
    
    for batch in tqdm(dataloader, desc="Computing gradient importance", disable=(get_rank() != 0)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        data_type = batch["data_type"][0]
        
        model.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                current_importance = (param.grad * param).abs().sum().item()
                importance[name] += current_importance
        
        if data_type == "pt":
            sample_counts["pt"] += 1
        else:
            sample_counts["sft"] += 1

        # Periodically clear GPU memory
        if get_rank() == 0 and sample_counts["pt"] % 100 == 0:
            torch.cuda.empty_cache()
    
    # Synchronize results across all processes (no-op in single-process mode)
    if dist.is_available() and dist.is_initialized():
        for name in importance:
            tensor = torch.tensor([importance[name]], device=device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            importance[name] = tensor.item()

        for key in sample_counts:
            count_tensor = torch.tensor([sample_counts[key]], device=device)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            sample_counts[key] = count_tensor.item()
    
    total_samples = sample_counts["pt"] + sample_counts["sft"]
    if total_samples > 0:
        for name in importance:
            importance[name] /= total_samples
    
    return importance

def print_sorted_importance(importance, top_k=None):
    """
    Print sorted importance scores.
    Args:
        importance: Dictionary of importance scores.
        top_k: Print only top-k results; if None, print all.
    """
    sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    if top_k is None:
        top_k = len(sorted_items)
    
    max_name_length = max(len(name) for name, _ in sorted_items[:top_k])
    
    print("\n" + "="*80)
    print("Gradient-based Parameter Importance Analysis")
    print("="*80)
    print(f"{'Parameter Name':<{max_name_length}} | {'Importance Score':>15} | {'Normalized Score':>15}")
    print("-"*80)
    
    max_importance = max(score for _, score in sorted_items)
    
    for name, score in sorted_items[:top_k]:
        normalized_score = score / max_importance
        print(f"{name:<{max_name_length}} | {score:15.6e} | {normalized_score:15.6f}")
    
    print("="*80)

def save_sorted_importance(importance, filepath):
    """
    Save sorted importance scores to a JSON file.
    """
    sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    max_importance = max(score for _, score in sorted_items)
    
    output_dict = {
        "raw_scores": dict(sorted_items),
        "normalized_scores": {
            name: score / max_importance 
            for name, score in sorted_items
        },
        "metadata": {
            "max_score": max_importance,
            "min_score": min(score for _, score in sorted_items),
            "num_parameters": len(sorted_items)
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(output_dict, f, indent=4)

def analyze_layer_importance(importance_dict):
    """
    Analyze layer-wise importance using two methods.
    """
    layer_info = {}
    for param_name, score in importance_dict.items():
        if 'model.layers.' not in param_name:
            continue
            
        parts = param_name.split('.')
        layer_num = int(parts[2])
        module_name = '.'.join(parts[3:])
        
        if layer_num not in layer_info:
            layer_info[layer_num] = {
                'params': {},
                'total_importance': 0.0,
                'rankings': []
            }
        
        layer_info[layer_num]['params'][module_name] = score
        layer_info[layer_num]['total_importance'] += score

    module_types = set()
    for layer_data in layer_info.values():
        module_types.update(layer_data['params'].keys())
    
    module_rankings = {module: {} for module in module_types}
    for module in module_types:
        scores = [(layer_num, layer_data['params'].get(module, 0.0)) 
                 for layer_num, layer_data in layer_info.items()]
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        total_layers = len(sorted_scores)
        for rank, (layer_num, _) in enumerate(sorted_scores, 1):
            normalized_rank = rank / total_layers
            layer_info[layer_num]['rankings'].append(normalized_rank)

    results = {
        'method1': {},
        'method2': {},
        'detailed': {}
    }

    max_importance = max(info['total_importance'] for info in layer_info.values())
    for layer_num, info in layer_info.items():
        normalized_importance = info['total_importance'] / max_importance
        results['method1'][f'layer_{layer_num}'] = normalized_importance

    for layer_num, info in layer_info.items():
        if info['rankings']:
            avg_ranking = sum(info['rankings']) / len(info['rankings'])
            results['method2'][f'layer_{layer_num}'] = 1 - avg_ranking

    for layer_num, info in layer_info.items():
        results['detailed'][f'layer_{layer_num}'] = {
            'total_importance': info['total_importance'],
            'normalized_importance': results['method1'][f'layer_{layer_num}'],
            'average_ranking': 1 - results['method2'][f'layer_{layer_num}'],
            'parameter_importance': info['params']
        }

    return results

def print_layer_importance(results):
    """
    Print layer-wise importance analysis results.
    """
    print("\n" + "="*100)
    print("Layer-wise Importance Analysis")
    print("="*100)
    
    layer_width = max(len(layer) for layer in results['method1'].keys())
    
    print(f"{'Layer':<{layer_width}} | {'Method 1 (Cumulative)':>20} | {'Method 2 (Ranking-based)':>20}")
    print("-"*100)
    
    sorted_layers = sorted(results['method1'].keys(), 
                         key=lambda x: results['method1'][x],
                         reverse=True)
    
    for layer in sorted_layers:
        method1_score = results['method1'][layer]
        method2_score = results['method2'][layer]
        print(f"{layer:<{layer_width}} | {method1_score:20.6f} | {method2_score:20.6f}")
    
    print("="*100)

def save_analysis_results(results, filepath):
    """
    Save layer-wise analysis results to a JSON file.
    """
    output_dict = {
        'methods': {
            'cumulative': results['method1'],
            'ranking_based': results['method2']
        },
        'detailed_analysis': results['detailed'],
        'statistics': {
            'num_layers': len(results['method1']),
            'method1_correlation_with_method2': compute_correlation(
                results['method1'].values(),
                results['method2'].values()
            )
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(output_dict, f, indent=4)

def compute_correlation(values1, values2):
    """
    Compute Pearson correlation between two sets of values.
    """
    values1 = list(values1)
    values2 = list(values2)
    return np.corrcoef(values1, values2)[0, 1]

class PruningImportanceAnalyzer:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.base_loss = None
        self.original_state = None

    def _compute_loss(self):
        total_loss = 0
        batch_count = 0
        
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.dataloader, 
                            desc="Computing loss",
                            disable=(get_rank() != 0)):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                if dist.is_initialized():
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / dist.get_world_size()
                
                total_loss += loss.item()
                batch_count += 1
                
                del outputs
                torch.cuda.empty_cache()
        
        if dist.is_initialized():
            count_tensor = torch.tensor([batch_count], device=self.device)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            batch_count = count_tensor.item()
        
        return total_loss / batch_count if batch_count > 0 else 0

    def compute_importance(self):
        self.original_state = copy.deepcopy(self.model.state_dict())
        
        if get_rank() == 0:
            print("Computing base loss on full dataset...")
        self.base_loss = self._compute_loss()
        
        importance_scores = {}
        layer_mapping = {}
        
        if get_rank() == 0:
            print("Analyzing layers...")
        
        for name, module in self.model.named_modules():
            if 'model.layers.' in name:
                parts = name.split('.')
                if len(parts) >= 3 and parts[1] == 'layers':
                    layer_num = int(parts[2])
                    if layer_num not in layer_mapping:
                        layer_mapping[layer_num] = []
                    layer_mapping[layer_num].append(name)

        for layer_num in tqdm(sorted(layer_mapping.keys()), 
                            desc="Evaluating layers",
                            disable=(get_rank() != 0)):
            layer_modules = layer_mapping[layer_num]
            
            saved_states = {}
            for module_name in layer_modules:
                module = self.model.get_submodule(module_name)
                saved_states[module_name] = {
                    name: param.data.clone()
                    for name, param in module.named_parameters()
                }
            
            for module_name in layer_modules:
                module = self.model.get_submodule(module_name)
                for param in module.parameters():
                    param.data.zero_()
            
            if dist.is_initialized():
                dist.barrier()
            
            if get_rank() == 0:
                print(f"\nComputing loss for layer {layer_num}...")
            pruned_loss = self._compute_loss()
            
            importance = pruned_loss - self.base_loss
            
            if dist.is_initialized():
                importance_tensor = torch.tensor([importance], device=self.device)
                dist.all_reduce(importance_tensor, op=dist.ReduceOp.SUM)
                importance = importance_tensor.item() / dist.get_world_size()
            
            importance_scores[f'layer_{layer_num}'] = importance
            
            for module_name in layer_modules:
                module = self.model.get_submodule(module_name)
                for name, param in module.named_parameters():
                    param.data.copy_(saved_states[module_name][name])
            
            del saved_states
            torch.cuda.empty_cache()
            
            if dist.is_initialized():
                dist.barrier()

        return importance_scores

def plot_importance_scores(importance_scores, save_path='path/to/your/pruning_importance.png'):
    plt.figure(figsize=(12, 6))
    
    layers = sorted(importance_scores.keys())
    scores = [importance_scores[layer] for layer in layers]
    
    min_score = min(scores)
    max_score = max(scores)
    normalized_scores = [(s - min_score) / (max_score - min_score) for s in scores]
    
    plt.plot(range(len(layers)), normalized_scores, 'b-', marker='o')
    plt.title('Layer Importance (Pruning-based Analysis)')
    plt.xlabel('Layer Index')
    plt.ylabel('Normalized Importance Score')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def print_importance_scores(importance_scores):
    print("\n" + "="*50)
    print("Layer Importance Analysis (Pruning-based)")
    print("="*50)
    
    min_score = min(importance_scores.values())
    max_score = max(importance_scores.values())
    normalized_scores = {
        layer: (score - min_score) / (max_score - min_score)
        for layer, score in importance_scores.items()
    }
    
    sorted_layers = sorted(normalized_scores.items(), 
                         key=lambda x: x[1], 
                         reverse=True)
    
    print(f"{'Layer':<15} | {'Normalized Score':>15} | {'Raw Score':>15}")
    print("-"*50)
    
    for layer, norm_score in sorted_layers:
        raw_score = importance_scores[layer]
        print(f"{layer:<15} | {norm_score:15.6f} | {raw_score:15.6f}")
    
    print("="*50)

def main():
    import argparse

    parser = argparse.ArgumentParser(description="ADEPT parameter/layer importance analysis.")
    parser.add_argument("--model_name_or_path", required=True)
    _default_data = str(
        (Path(__file__).resolve().parents[1] / "shared/data/adept_general_competence.json")
    )
    parser.add_argument(
        "--data_path",
        default=_default_data,
        help="JSON list with `text` or instruction/output fields. "
        "Default: shared/data/adept_general_competence.json (ADEPT paper Appendix B.3).",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument(
        "--top_k_expand",
        type=int,
        default=4,
        help="Suggest this many lowest-importance layers to expand "
        "(paper B.7 main setting: k=4; Llama medical exception used k=8).",
    )
    args = parser.parse_args()

    if not Path(args.data_path).is_file():
        raise FileNotFoundError(
            f"Importance data not found: {args.data_path}. "
            "Run: python shared/scripts/prepare_adept_general_competence.py"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    local_rank, num_gpus = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if local_rank == 0:
        print(f"Using {num_gpus} GPUs")
        print(f"model={args.model_name_or_path}")
        print(f"data={args.data_path}")
        print(f"output_dir={args.output_dir}")

    if dist.is_initialized():
        dist.barrier()

    model_name = args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank} if num_gpus == 1 else "balanced",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if dist.is_initialized():
        dist.barrier()

    if local_rank == 0 or not (dist.is_available() and dist.is_initialized()):
        with open(args.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    if dist.is_available() and dist.is_initialized():
        if local_rank == 0:
            data_str = json.dumps(data)
            data_bytes = data_str.encode("utf-8")
            data_tensor = torch.ByteTensor(list(data_bytes)).to(device)
            data_size = torch.LongTensor([len(data_bytes)]).to(device)
        else:
            data_size = torch.LongTensor([0]).to(device)

        dist.broadcast(data_size, src=0)
        if local_rank != 0:
            data_tensor = torch.ByteTensor([0] * data_size.item()).to(device)

        dist.broadcast(data_tensor, src=0)

        if local_rank != 0:
            data_bytes = bytes(data_tensor.cpu().tolist())
            data_str = data_bytes.decode("utf-8")
            data = json.loads(data_str)

    dataset = SimpleDataset(data, tokenizer, max_length=args.max_length)
    dataloader, sampler = create_dataloader(dataset, batch_size=args.batch_size, local_rank=local_rank)

    importance = gradient_importance_analysis(model, dataloader, device)

    pruning_dataloader, _ = create_dataloader(
        dataset, batch_size=max(1, args.batch_size // 2), local_rank=local_rank
    )
    pruning_analyzer = PruningImportanceAnalyzer(model, pruning_dataloader, device)
    pruning_importance = pruning_analyzer.compute_importance()

    if local_rank == 0:
        param_out = os.path.join(args.output_dir, "param_importance_sorted.json")
        layer_out = os.path.join(args.output_dir, "layer_importance_sorted.json")
        prune_png = os.path.join(args.output_dir, "pruning_importance.png")
        prune_json = os.path.join(args.output_dir, "pruning_importance_analysis.json")
        expand_out = os.path.join(args.output_dir, "suggested_expand_layers.txt")

        print_sorted_importance(importance, top_k=None)
        save_sorted_importance(importance, param_out)

        layer_results = analyze_layer_importance(importance)
        print_layer_importance(layer_results)
        save_analysis_results(layer_results, layer_out)

        ranked = sorted(layer_results["method1"].items(), key=lambda x: x[1])
        expand_ids = [int(name.split("_")[1]) for name, _ in ranked[: args.top_k_expand]]
        expand_ids_sorted = sorted(expand_ids)
        with open(expand_out, "w", encoding="utf-8") as f:
            f.write(",".join(str(i) for i in expand_ids_sorted) + "\n")
        print(f"Suggested expand_layers (lowest general importance, k={args.top_k_expand}): {expand_ids_sorted}")
        print(f"Wrote {expand_out}")

        print_importance_scores(pruning_importance)
        plot_importance_scores(pruning_importance, prune_png)

        with open(prune_json, "w", encoding="utf-8") as f:
            min_val = min(pruning_importance.values())
            max_val = max(pruning_importance.values())
            json.dump(
                {
                    "raw_scores": pruning_importance,
                    "normalized_scores": {
                        layer: (score - min_val) / (max_val - min_val) if max_val > min_val else 0.0
                        for layer, score in pruning_importance.items()
                    },
                },
                f,
                indent=4,
            )

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
