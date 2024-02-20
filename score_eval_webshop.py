import argparse, json, os, re
from tqdm import tqdm
from typing import Dict, List
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import adaptive_utils

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="codellama/CodeLlama-7b-Instruct-hf", help="Model name")
parser.add_argument("--cache_dir", type=str, default="", help="Cache directory for the model")
parser.add_argument("--load_in_4bit", type=bool, default=True, help="Load in 4-bit")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--data_path", type=str, help="Path to the data")
parser.add_argument("--output_dir", type=str, help="Output directory")
parser.add_argument("--aggregation", type=str, default="mean", help="Aggregation method (mean, min, prod)")
parser.add_argument("--method", type=str, default="SC", help="Method (SC, SoftSC, AdaptiveSC, AdaptiveSoftSC)")
parser.add_argument("--threshold", type=float, default=0.8, help="Threshold for AdaptiveSC and AdaptiveSoftSC")
args = parser.parse_args()

args.device = "cuda" if torch.cuda.is_available() else "cpu"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

transformers.set_seed(args.seed)

tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
if args.load_in_4bit:
    model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir=args.cache_dir, device_map="auto", quantization_config=bnb_config)
else:
    model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir=args.cache_dir, device_map="auto",
                                                 torch_dtype=torch.float16 if args.device == "cuda" else torch.bfloat16)

tokenizer.pad_token_id = tokenizer.eos_token_id

def find_start(tokens):
    match_token = ['<0x0A>', 'Action', ':']
    for i in range(len(tokens) - 3, 0, -1):
        if tokens[i : i + 3] == match_token:
            break
    return i + 3

softmax = torch.nn.Softmax(dim=1)
@torch.no_grad()
def get_probs(input):
    input = tokenizer(input, return_tensors="pt").to(args["device"])
    input_tokens = tokenizer.convert_ids_to_tokens(input["input_ids"][0])
    output = model(**input, return_dict=True)
    logits = output.logits[0].to("cpu")
    probs = softmax(logits)
    start = find_start(input_tokens)
    probs_list = [probs[i-1][input["input_ids"][0][i]].item() for i in range(start, len(input_tokens)-1)]
    probs_list_np = np.array(probs_list)

    return probs_list_np

with open(args.data_path, "r") as f:
    data = json.load(f)
results_path = os.path.join(args.output_dir, "results.json")

prompts = data["prompts"]

for key in data:
    if key == "prompts":
        continue
    problem = data[key]["problem"]
    products = data[key]["products"]
    scores = data[key]["scores"]
    rewards = data[key]["reward"]
    run_keys = []
    for subkey in data[key]:
        if subkey.startswith("run"):
            run_keys.append(subkey)

    init_prompt_select = prompts["select"].format(problem)

    for i in range(len(run_keys)):
        act_history = data[key][run_keys[i]]["act_history"]
        trace = data[key][run_keys[i]]["trace"]
        prompt = init_prompt_select
        j = 0
        probs_list = []
        avg_probs_list = []
        lowest_probs_list = []
        prod_logprobs_list = []
        for act in act_history["select"]:
            if act == "click[Back to Search]":
                break
            while act not in trace["select"][j]:
                prompt += "\n" + trace["select"][j]
                j += 1
            input = prompt + "\n\nAction: " + act + "\n"
            logprobs = get_probs(input)
            probs_list.append(logprobs.tolist())
            avg_probs_list.append(np.mean(logprobs))
            lowest_probs_list.append(np.min(logprobs))
            prod_logprobs_list.append(np.mean(np.log(logprobs)))
        prompt = prompts["buy"].format(products[i], problem)
        j = 0
        for act in act_history["buy"]:
            while act not in trace["buy"][j]:
                prompt += "\n" + trace["buy"][j]
                j += 1
            input = prompt + "\n\nAction: " + act + "\n"
            logprobs = get_probs(input)
            probs_list.append(logprobs.tolist())
            avg_probs_list.append(np.mean(logprobs))
            lowest_probs_list.append(np.min(logprobs))
            prod_logprobs_list.append(np.mean(np.log(logprobs)))
        data[key][run_keys[i]]["probs_list"] = probs_list
        data[key][run_keys[i]]["avg_probs_list"] = avg_probs_list
        data[key][run_keys[i]]["lowest_probs_list"] = lowest_probs_list
        data[key][run_keys[i]]["prod_logprobs_list"] = prod_logprobs_list
            
    with open(results_path, "w") as f:
        json.dump(data, f, indent=2)

data = {}
with open(results_path, "r") as f:
    fdata = json.load(f)
for key in fdata:
    if key.startswith("fixed_"):
        data[key] = fdata[key]

success_num = 0
total_score = 0
for key in data:
    scores = data[key]["scores"]
    rewards = data[key]["reward"]
    products = data[key]["products"]
    run_keys = []
    for subkey in data[key]:
        if subkey.startswith("run"):
            run_keys.append(subkey)
    if args.method == "SC":
        major_product = max(set(products), key=products.count)
        idx = products.index(major_product)
        total_score += scores[idx]
        if rewards[idx]:
            success_num += 1
    elif args.method == "SoftSC":
        if args.aggregation == "mean":
            probs_list = [np.mean(np.array(data[key][run_key]["avg_probs_list"])) for run_key in run_keys]
        elif args.aggregation == "min":
            probs_list = [np.min(np.array(data[key][run_key]["lowest_probs_list"])) for run_key in run_keys]
        elif args.aggregation == "prod":
            probs_list = [np.mean(np.array(data[key][run_key]["prod_logprobs_list"])) for run_key in run_keys]
        total_score += scores[np.argmax(probs_list)]
        if rewards[np.argmax(probs_list)]:
            success_num += 1
    elif args.method == "AdaptiveSC":
        adaptive_products = adaptive_utils.adptive_sampling(products, threshold=args.threshold)
        major_product = max(set(adaptive_products), key=adaptive_products.count)
        idx = products.index(major_product)
        total_score += scores[idx]
        if rewards[idx]:
            success_num += 1
    elif args.method == "AdaptiveSoftSC":
        lowest_probs_list = [np.mean(np.array(data[key][run_key]["lowest_probs_list"])) for run_key in run_keys]
        generations = []
        probs_sum = 0.0
        j = 0
        while probs_sum < args.threshold and len(generations) < len(products):
            probs_sum += lowest_probs_list[j]
            generations.append((j, products[j], lowest_probs_list[j]))
            j += 1

        idx, best_generation, best_prob = sorted(generations, key=lambda x: x[2])[-1]

        total_score += scores[idx]
        if rewards[idx]:
            success_num += 1

print(f"Success rate: {success_num / len(data)}, Average score: {total_score / len(data)}")
            
