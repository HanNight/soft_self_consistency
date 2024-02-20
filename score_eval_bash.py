import argparse, json, os, re
from tqdm import tqdm
from typing import Dict, List
import argparse

import random
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from prompts.prompts import TemplateV2
import adaptive_utils

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, default="codellama/CodeLlama-7b-Instruct-hf", help="Model name")
parser.add_argument("--cache_dir", type=str, default="", help="Cache directory for the model")
parser.add_argument("--load_in_4bit", type=bool, default=True, help="Load in 4-bit")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--k", type=int, default=20, help="Number of self consistency runs")
parser.add_argument("--input_file", type=str, default="", help="The path to the input file for scoring and selection")
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

def bash_parser(action: str):
    # action = eval(action)
    action = re.sub("\n", " ", action)
    if '```' not in action:
        if action.lower().startswith(f"bash: "):
            action = action[len(f"bash: "):]
            return action, True
        if "command:" in action.lower():
            action = action[action.lower().index("command:") + len("command:"):]
            return action, True

    pattern1 = f'```(?:bash|BASH|sh)?([\S\s]+?)```'
    pattern2 = f'```([\S\s]+?)```'
    matches = re.findall(pattern1, action.lower(), re.DOTALL) + re.findall(pattern2, action.lower(), re.DOTALL)
    if len(matches) == 0:
        return action, True
    action = " ".join(matches[0].split())
    return action, True

def llama_encode_message(dialogue):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    if dialogue[0]["role"] == "system":
        dialogue = [
                {
                    "role": dialogue[1]["role"],
                    "content": f"{B_SYS} {dialogue[0]['content']} {E_SYS} {dialogue[1]['content']}",
                }
            ] + dialogue[2:]

    chat = ""
    for d in dialogue:
        if d["role"] == "user":
            chat += f"{B_INST} {d['content']} {E_INST} "
        elif d["role"] == "assistant":
            chat += d['content'] + "\n"
    return chat

stop_sequences_for_action = ["[INST]", "[/INST]", "Observation:", "Task:", "Output:", "Checklist", "Explanantion:", "Solution:"]
post_process_func = lambda x: x.split("[/INST]")[0].strip()

language = "bash"
setting = "Bourne Shell"
template = TemplateV2(language, setting)

def model_chat(chat, stop_sequences=None, max_new_tokens=1024):
    input = tokenizer(chat, return_tensors="pt").to(args.device)
    output = model.generate(**input,
                            temperature=0.7,
                            do_sample=True,
                            max_new_tokens=max_new_tokens, 
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            )
    output = output.to("cpu")
    result = tokenizer.decode(output[0][input["input_ids"].shape[1]:], skip_special_tokens=True)
    for stop_sequence in stop_sequences:
        if stop_sequence in result:
            result = result[:result.index(stop_sequence)]
    return result

def single_turn_dialogue(query, action=None):
    dialogue = [{"role": "system", "content": f"You are a helpful assistant expert specializing in {language}."}]
    dialogue += [{"role": "user", "content": template.get_init_msg() + "\n" + template.get_query_msg(query)}]
    if action is not None:
        dialogue.append({"role": "assistant", "content": f"```{language.lower()}\n{action}\n```"})
    return dialogue

def find_start(tokens):
    match_token = ['▁```', 'bash', '<0x0A>']
    for i in range(len(tokens) - 3, 0, -1):
        if tokens[i : i + 3] == match_token:
            break
    return i + 3

softmax = torch.nn.Softmax(dim=1)
@torch.no_grad()
def get_probs(input):
    input = tokenizer(input, return_tensors="pt").to(args.device)
    input_tokens = tokenizer.convert_ids_to_tokens(input["input_ids"][0])
    output = model(**input, return_dict=True)
    logits = output.logits[0].to("cpu")
    probs = softmax(logits)
    start = find_start(input_tokens)
    probs_list = [probs[i-1][input["input_ids"][0][i]].item() for i in range(start, len(input_tokens)-3)]
    probs_list_np = np.array(probs_list)

    return probs_list_np

input_file_name = os.path.split(os.path.splitext(args.input_file)[0])[-1]

results_path = os.path.join(args.output_dir, f"bash_{input_file_name}_score_{args.method}_{args.aggregation}.json")


with open(args.input_file, "r") as f:
    data = json.load(f)

for key in data:
    query = data[key]["query"]
    actions = data[key]["actions"]

    probs_list = []
    avg_probs_list = []
    lowest_probs_list = []
    prod_logprobs_list = []
    for action in actions:
        dialogue = single_turn_dialogue(query, action)
        chat = llama_encode_message(dialogue)
        probs = get_probs(chat)
        probs_list.append(probs.tolist())
        avg_probs_list.append(np.mean(probs))
        lowest_probs_list.append(np.min(probs))
        prod_logprobs_list.append(np.mean(np.log(probs)))
    data[key]["probs_list"] = probs_list
    data[key]["avg_probs_list"] = avg_probs_list
    data[key]["lowest_probs_list"] = lowest_probs_list
    data[key]["prod_logprobs_list"] = prod_logprobs_list

    selected_idx = 0
    if args.method == "SC":
        major_action = max(set(actions), key=actions.count)
        selected_idx = actions.index(major_action)
    elif args.method == "SoftSC":
        if args.aggregation == "mean":
            probs_list = np.array(data[key]["avg_probs_list"])
        elif args.aggregation == "min":
            probs_list = np.array(data[key]["lowest_probs_list"])
        elif args.aggregation == "prod":
            probs_list = np.array(data[key]["prod_logprobs_list"])
        selected_idx = int(np.argmax(probs_list))
    elif args.method == "AdaptiveSC":
        adaptive_actions = adaptive_utils.adptive_sampling(actions, threshold=args.threshold)
        major_action = max(set(adaptive_actions), key=adaptive_actions.count)
        selected_idx = actions.index(major_action)
    elif args.method == "AdaptiveSoftSC":
        avg_probs_list = np.array(data[key]["avg_probs_list"][:10])
        lowest_probs_list = np.array(data[key]["lowest_probs_list"][:10])
        generations = []
        probs_sum = 0.0
        j = 0
        while probs_sum < args.threshold and len(generations) < len(actions):
            probs_sum += lowest_probs_list[j]
            generations.append((j, actions[j], avg_probs_list[j]))
            j += 1
        idx, best_generation, best_prob = sorted(generations, key=lambda x: x[2])[-1]
        selected_idx = idx
    
    data[key]["selected_idx"] = selected_idx

with open(results_path, "w") as f:
    json.dump(data, f, indent=2)