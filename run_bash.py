import argparse, json, os, re
from intercode.envs import (
    BashEnv, CTFEnv, PythonEnv, SqlEnv, ACTION_EXEC, AGENT_OBS
)
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

data_path_template = "data/bash/nl2bash/nl2bash_fs_{}.json"
image_name_template = "intercode-nl2bash{}"

results_path = os.path.join(args.output_dir, f"bash_self-consistency_{args.model}_sample{args.k}_seed{args.seed}.json")

results = {}

for n in range(1, 5):
    results[n] = {}
    env = BashEnv(image_name=image_name_template.format(n), data_path=data_path_template.format(n))

    for i in tqdm(range(0, len(env.data_loader))):
        env.reset(i)
        results[n][i] = {"query": env.query, "gold": env.gold}

        query = env.query
        actions = []
        valid_action = []
        rewards = []
        for j in range(args.k):
            dialogue = single_turn_dialogue(query)
            chat = llama_encode_message(dialogue)
            result = model_chat(chat, stop_sequences_for_action, 512)
            result = post_process_func(result)
            action = result[0] if isinstance(result, list) else result
            action, is_code = bash_parser(action)
            actions.append(action)
            if not is_code:
                rewards.append(0)
                valid_action.append(False)
            else:
                env.reset(i)
                observation, reward, done, info = env.step(action)
                valid_action.append(info[ACTION_EXEC])
                try:
                    _, reward, done, info = env.step("submit")
                    rewards.append(reward)
                except:
                    rewards.append(-1)
        results[n][i]["actions"] = actions
        results[n][i]["valid_action"] = valid_action
        results[n][i]["rewards"] = rewards

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
        results[n][i]["probs_list"] = probs_list
        results[n][i]["avg_probs_list"] = avg_probs_list
        results[n][i]["lowest_probs_list"] = lowest_probs_list
        results[n][i]["prod_logprobs_list"] = prod_logprobs_list

        if i % 5 == 0:
            with open(results_path, "w") as fp:
                json.dump(results, fp, indent=2)

    with open(results_path, "w") as fp:
        json.dump(results, fp, indent=2)
    env.close()

with open(results_path, "w") as fp:
    json.dump(results, fp, indent=2)

total_data_num = 0
success_num = 0
for n in results:
    for i in results[n]:
        total_data_num += 1
        actions = results[n][i]["actions"]
        rewards = results[n][i]["rewards"]
        if args.method == "SC":
            major_action = max(set(actions), key=actions.count)
            if rewards[actions.index(major_action)] == 1:
                success_num += 1
        elif args.method == "SoftSC":
            if args.aggregation == "mean":
                probs_list = np.array(results[n][i]["avg_probs_list"])
            elif args.aggregation == "min":
                probs_list = np.array(results[n][i]["lowest_probs_list"])
            elif args.aggregation == "prod":
                probs_list = np.array(results[n][i]["prod_logprobs_list"])
            if rewards[np.argmax(probs_list)] == 1:
                success_num += 1
        elif args.method == "AdaptiveSC":
            adaptive_actions = adaptive_utils.adptive_sampling(actions, threshold=args.threshold)
            major_action = max(set(adaptive_actions), key=adaptive_actions.count)
            if rewards[actions.index(major_action)] == 1:
                success_num += 1
        elif args.method == "AdaptiveSoftSC":
            avg_probs_list = np.array(results[n][i]["avg_probs_list"][:10])
            lowest_probs_list = np.array(results[n][i]["lowest_probs_list"][:10])
            generations = []
            probs_sum = 0.0
            j = 0
            while probs_sum < args.threshold and len(generations) < len(actions):
                probs_sum += lowest_probs_list[j]
                generations.append((j, actions[j], avg_probs_list[j]))
                j += 1
            idx, best_generation, best_prob = sorted(generations, key=lambda x: x[2])[-1]
            if rewards[idx] == 1:
                success_num += 1
print(f"Success Rate: {success_num / total_data_num}")