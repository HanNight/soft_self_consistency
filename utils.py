from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
import torch
import numpy as np

local_dir = ".cache/hub/models--codellama--CodeLlama-34b-Instruct-hf/snapshots/"
tokenizer = AutoTokenizer.from_pretrained(f'{local_dir}')

tokenizer.pad_token_id = tokenizer.eos_token_id    # for open-ended generation

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    f'{local_dir}',
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True)

softmax = torch.nn.Softmax(dim=1)

generation_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    trust_remote_code=True,
    device_map="auto",    # finds GPU
)

def opensource_completion(prompt,max_tokens, k=1, stop='\n'):
    temp = 0.7 if k == 1 else 1
    top_p = 1 if k == 1 else 0.9
    top_k = 40
    sequences = generation_pipe(
        prompt,
        temperature = temp,
        max_new_tokens=max_tokens,
        num_return_sequences = k,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_full_text=False,
        top_p = top_p,
        do_sample = True,
    )
    if k == 1:
        return sequences[0]["generated_text"].split(stop)[0]
    else:
        return [seq["generated_text"].split(stop)[0] for seq in sequences]

def find_start(tokens):
    match_token = ['>']
    for i in range(len(tokens) - 1, 0, -1):
        if tokens[i : i + 1] == match_token:
            break
    return i + 1

@torch.no_grad()
def get_probs(input):
    input = tokenizer(input, return_tensors="pt").to('cuda')
    input_tokens = tokenizer.convert_ids_to_tokens(input["input_ids"][0])
    output = model(**input, return_dict=True)
    logits = output.logits[0].to("cpu")
    probs = softmax(logits)
    start = find_start(input_tokens)
    probs_list = [probs[i-1][input["input_ids"][0][i]].item() for i in range(start, len(input_tokens))]
    probs_list_np = np.array(probs_list)

    return probs_list_np

def score_select(actions, prompt, initial_prompt, mode='avg'):
    probs_list = []
    avg_probs_list = []
    lowest_probs_list = []
    prod_probs_list = []
    if isinstance(prompt, str): prompt = [prompt] * len(actions)
    for a, action in enumerate(actions):
        # this is the input to get the probs
        input = initial_prompt + prompt[a] + "\n> " + action
        probs = get_probs(input)
        probs_list.append(probs.tolist())
        avg_probs_list.append(np.mean(probs))
        lowest_probs_list.append(np.min(probs))
        prod_probs_list.append(np.mean(np.log(probs)))
    if mode == 'avg':
    # using avg probs:
        best_action = actions[np.argmax(avg_probs_list)]
    # using lowest probs:
    elif mode == 'min':
        best_action = actions[np.argmax(lowest_probs_list)]
    # using length normalized product of probs:
    elif mode == 'prod':
        best_action = actions[np.argmax(prod_probs_list)]
    else: assert False, 'Invalid scoring function'
    return best_action
