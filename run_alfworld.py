import os
import argparse
import openai
import json
import sys
from tqdm import tqdm
import copy
import gym
import random
import re
import yaml
import alfworld
import alfworld.agents.environment
from tenacity import (
    retry,
    stop_after_attempt, # type: ignore
    wait_random_exponential, # type: ignore
)
import alf_utils as utils
from alf_utils import score_select
openai.api_key = os.environ['OPENAI_KEY']
os.environ["ALFWORLD_DATA"] = 'alfworld/data'
import warnings
warnings.filterwarnings("ignore")
import transformers
parser = argparse.ArgumentParser()
parser.add_argument("--LM", help="Name of OpenAI language model to be used. (by default planner)", type=str, default='llama')
parser.add_argument("--exec-LM", help="Name of the Executor LLM to be used.", type=str, default='llama')
parser.add_argument('--force-same-LM', help="force planner and executor LM to be the same.", type=bool, default=True)
parser.add_argument("--eval-type", help='Choose between in-domain or ood eval set (former is val set)', choices=['id', 'ood'], default='ood')
parser.add_argument("--fname", help='destination filename keyword for saving traces/logs', type=str)
parser.add_argument("--score", help='Type of scoring function', type=str, default='avg', choices=['min', 'prod', 'avg', 'majority'])
parser.add_argument('--num-task-samples', help='How much subsampling per task if at all?', default=5, type=int)
parser.add_argument('--eval-all', help='Should we evaluate on the full test set?', default=False, type=bool)
parser.add_argument('--max-depth', help='Max decomposition depth', default=1, type=int)
parser.add_argument('--k', help='Self-consistency over k samples', default=5, type=int)
parser.add_argument('--seed', help='Seed for generation', default=0, type=int)
parser.add_argument('--max-runs', help='Max number of commands that can be executed in one run.', default=30, type=int)
parser.add_argument('--executor', help='Type of executor to be used', default='react', choices=['react', 'atomic', 'hybrid'])
parser.add_argument('--react-type', help='If using react exec, type of fs prompt to use.', default='std', choices=['std', 'cross', 'common'])
parser.add_argument('--store-results', help="Write the results in a separate file.", type=bool, default=True)
parser.add_argument('--adaptive', help="Implement the adaptive version", type=bool, default=False)
parser.add_argument('--no-store', help="Write the results in a separate file.", dest='store_results', action='store_false')
parser.add_argument('--info-prop-mode', help="What format of information should be propogated across executors?", default='last-step-last-act', choices=['last-step-last-act', 'all-step-last-act'])
parser.add_argument('--verbose', help='Verbosity: Print intermediate outputs.', default=False, type=bool)
parser.add_argument('--system-seed', help='Set common system independent seed', default=True, type=bool)
args = parser.parse_args()

transformers.set_seed(args.seed)
if args.adaptive:
    from alf_utils import adaptive_score_select as score_select


LM = args.LM
if args.force_same_LM: 
    exec_LM = args.LM
else:
    exec_LM = args.exec_LM

max_runs = args.max_runs
environment_context = 'List of viable commands:\n\
- go to {recep}\n\
- open {recep}\n\
- close {recep}\n\
- take {obj} from {recep}\n\
- put {obj} in/on {recep}\n\
- use {lamp}\n\
- look\n\
- inventory\n\
- heat {obj} with {microwave}\n\
- cool {obj} with {fridge}\n\
- clean {obj} with {sink}\n\
- clean {obj} with {bathtub}\n\
- slice {obj} with {knife}\n\
where, "{recep}" denotes a receptacle, "{obj}" denotes any object from the environment, etc. Use the full name of the objects/receptacles including identifying number based on previous observations. Commands "heat", "cool", and "clean" are high-level shortcuts, so you should ONLY use "go to" commands before them. DO NOT use "open"/"put"/"take" commands if you plan to use "clean"/"heat"/"cool". Do not make assumptions if an object is hot/clean/cool without verifying.'


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
def llm(prompt, stop=["\n"], k=args.k):
    # print(exec_LM)
    if 'davinci' in exec_LM: # or 'turbo-instruct' in LM:
      if isinstance(prompt, list): prompt = prompt[0]
      response = openai.Completion.create(
        model=exec_LM,
        prompt='Interact with a household to solve a task. ' + 'Commands "heat", "cool", and "clean" are high-level shortcuts, so you should only use "go to" commands before them. Do not use "open"/"put"/"take" commands if you plan to use "clean"/"heat"/"cool". Do not make assumptions if an object is hot/clean/cool without verifying.\n' + prompt,
        # 'Note that commands "heat", "cool", and "clean" are high-level actions, so you can only use "go to" commands before them. Do not use "open"/"put"/"take" before high-level actions.\n' + prompt,
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop
      )
      return response["choices"][0]["text"]
    
    elif 'llama' in exec_LM.lower():
        if isinstance(prompt, list): prompt = prompt[0]
        prompt= 'Interact with a household to solve a task. ' + 'Commands "heat", "cool", and "clean" are high-level shortcuts, so you should only use "go to" commands before them. Do not use "open"/"put"/"take" commands if you plan to use "clean"/"heat"/"cool". Do not make assumptions if an object is hot/clean/cool without verifying.\n' + prompt
        response = utils.opensource_completion(prompt, 100, k=k)
        return response

    
    elif 'turbo-instruct' in exec_LM:
        if isinstance(prompt, list): prompt = prompt[0]
        response = openai.Completion.create(
            model=exec_LM,
            prompt='Interact with a household to solve a task.\n' + environment_context + '\n' + prompt,
            temperature=0,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop
      )
        return response["choices"][0]["text"]

    elif 'gpt-3.5' in LM or 'gpt-4' in exec_LM: # models that support "chat"
          
      assert isinstance(prompt, list), print("Incorrect prompt format, expecting list of dict (messages)")
      messages = [
          {"role": "system", "content": 'You are a helpful robot navigating through a household. Interact with a household to solve a task by telling me the next action. Actions can be commands for the environment or thoughts/comments. All thoughts/comments or non-valid commands always start with "think: ". Actions will be passed to the environment which will return observations. Choose actions based on the observations.\n' + environment_context + '\n'}
	      ]  
      for chat in prompt:
          if chat['type'] == 'act':
            messages.append({'role': 'assistant', 'content': chat['content']})
          else:
            messages.append({'role': 'user', 'content': chat['content']})
      response = openai.ChatCompletion.create(
        model=exec_LM,
        messages=messages,
         
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop
      )
      choices = response["choices"]
      completion_objs = [choice.message for choice in choices]
      completions = [completion.content for completion in completion_objs]
      return completions[0]
    
    
with open('alfworld/configs/base_config.yaml') as reader:
    config = yaml.safe_load(reader)

if args.eval_type == 'ood':    
    split = "eval_out_of_distribution"
elif args.eval_type == 'id':
    split = "eval_in_distribution"
else: assert False, 'Entered eval_type is incorrect!'

orig_env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
game_files = orig_env.game_files
if args.system_seed:
    game_files.sort()

env = orig_env.init_game(batch_size=1, game_file=game_files[2])

def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]  
        ob = 'You arrive at the location. '  + ob
    return ob

folder = './prompts/'
prompt_file = 'alfworld_3prompts_endings.json'
plan_prompt_file = 'alfworld_plan_filled_prompts.json'
if 'davinci' in exec_LM:
    atomic_exec_file = 'alfworld_atomic_exec_prompts.json'
else: atomic_exec_file = 'alfworld_atomic_exec_prompts.json' #Done for turbo-instruct

if args.store_results: 
    os.makedirs(f"results/alfworld/Mistrl7B_{LM}/", exist_ok=True) 
    dest_file = './results/alfworld/Mistrl7B_{}/{}_{}.json'.format(LM,args.fname,LM)
    t = open(dest_file, 'w+')
with open(folder + prompt_file, 'r') as f:
    d = json.load(f)
with open(folder + plan_prompt_file, 'r') as f:
    pdict = json.load(f)
with open(folder + atomic_exec_file, 'r') as f:
    exdict = json.load(f)

prefixes = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}

ordered_game_files = {v:[] for v in prefixes.values()}
for game in game_files:
    for k,v in prefixes.items():
        if k in game: ordered_game_files[v].append(game)

random.seed(0)
sel_game_files = []
if not args.eval_all:
    for lst in ordered_game_files.values():
        sel_game_files.extend(random.sample(lst, k=args.num_task_samples))
else:
    for lst in ordered_game_files.values():
        sel_game_files.extend(lst)
print('Selected {} tasks from {}'.format(len(sel_game_files), split))

def convert_messages(prompt):
    messages = []
    entry = prompt.split('> ')
    messages.append({'type': 'env', 'content': entry[0]})
    for item in entry[1:]:
        item = item.rstrip('\n')
        cmmds = item.split('\n')
        if len(cmmds) > 1:
            act = cmmds[0]
            env = cmmds[1]
            if not len(env): env = 'OK.'
        else:
            act = cmmds[0]
            env = None
        messages.append({'type': 'act', 'content': act})
        if not env is None: messages.append({'type': 'env', 'content': env})
    return messages

def fetch_obj_recept(filename):
    entities = filename.split('-')
    if entities[2] != 'None':
        return [entities[1].lower(), entities[2].lower()], entities[3].lower()
    return [entities[1].lower(), ''], entities[3].lower()


if exec_LM in ["gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo-0301"]:
    atomic_exec_prompt = [{'type':'env', 'content': 'Here is a demo of actions you can perform.\n' + exdict['room'] + "\n"}]
    for i in range(len(exdict.keys()) -1):
        atomic_exec_prompt.extend(convert_messages(exdict['action_{}'.format(str(i))]))
    # atomic_exec_prompt.append({"type": "env", "content": 'Commands "heat", "cool", and "clean" are high-level actions. Therefore, use only "go to" command followed by "clean"/"heat"/"cool". Do not use "open"/"put"/"take" commands directly before "clean"/"heat"/"cool" commands.'})
    atomic_exec_prompt.append({'type':'env', 'content':'Here is a complex task for you to perform. If you think you have finished the task, think "Task completed!". If after trying your best, you conclude that the task is infeasible, think "Task failed!". Always start with a "think" statement.'})
else:
    atomic_exec_prompt = 'Here is a demo of actions you can perform.\n\n' + exdict['room'] + "\n\n"
    for i in range(len(exdict.keys()) -1):
        atomic_exec_prompt += exdict['action_{}'.format(str(i))] + '\n\n'
    atomic_exec_prompt += 'Here is a complex task for you to perform. If you think you have finished the task, think "Task completed!". If after trying your best, you conclude that the task is infeasible, think "Task failed!".'

def fetch_react_prompt(idx, prefixes, d, type):
    if type == 'std':
        v = prefixes[idx]
        if 'davinci' in exec_LM or 'turbo-instruct' in exec_LM or 'llama' in exec_LM.lower():
            # prompt = 'Here are two examples.\n' + d[f'react_{v}_1'] + d[f'react_{v}_0'] + '\nHere is the task. If you think you have finished the task, think "Task completed!". If after trying your best, you conclude that the task is infeasible, think "Task failed!\n'
            prompt = 'Here are two examples.\n' + d[f'react_{v}_1'] + '\nHere is the task. If you think you have finished the task, think "Task completed!". If after trying your best, you conclude that the task is infeasible, think "Task failed!\n'

        elif 'gpt-3.5' in exec_LM or 'gpt-4' in exec_LM:
            prompt = [{'type': 'env', 'content': 'Here are two examples.'}] + convert_messages(d[f'react_{v}_1']) + convert_messages(d[f'react_{v}_0']) + [{'type':'env', 'content': 'Here is the task. If you think you have finished the task, think "Task completed!". If after trying your best, you conclude that the task is infeasible, think "Task failed!'}]      
    elif type == 'cross':
        alt_indices = [k for k in prefixes.keys() if k!=idx]
        v1 = prefixes[random.sample(alt_indices, k=1)[0]]
        v2 = prefixes[random.sample(alt_indices, k=1)[0]]
        if 'davinci' in exec_LM or 'turbo-instruct' in exec_LM or 'llama' in exec_LM.lower():
            v = prefixes[idx]
            prompt = 'Here are two examples.\n' + d[f'react_{v1}_1'] + d[f'react_{v2}_0'] + '\nHere is the task. If you think you have finished the task, think "Task completed!". If after trying your best, you conclude that the task is infeasible, think "Task failed!\n'
        elif 'gpt-3.5' in exec_LM or 'gpt-4' in exec_LM:
            prompt = [{'type': 'env', 'content': 'Here are two examples.'}] + convert_messages(d[f'react_{v1}_1']) + convert_messages(d[f'react_{v2}_0']) + [{'type':'env', 'content': 'Here is the task. If you think you have finished the task, think "Task completed!". If after trying your best, you conclude that the task is infeasible, think "Task failed!'}]      
    elif type == 'common':
        v1 = 'heat'
        v2 = 'examine'
        if 'davinci' in exec_LM or 'turbo-instruct' in exec_LM or 'llama' in exec_LM.lower():
            v = prefixes[idx]
            prompt = 'Here are two example tasks.\n' + d[f'react_{v1}_1'] + d[f'react_{v2}_0'] + '\nHere is the task. If you think you have finished the task, think "Task completed!". If after trying your best, you conclude that the task is infeasible, think "Task failed!\n'
        elif 'gpt-3.5' in exec_LM or 'gpt-4' in exec_LM:
            prompt = [{'type': 'env', 'content': 'Here are two example tasks.'}] + convert_messages(d[f'react_{v1}_1']) + convert_messages(d[f'react_{v2}_0']) + [{'type':'env', 'content': 'Here is the task. If you think you have finished the task, think "Task completed!". If after trying your best, you conclude that the task is infeasible, think "Task failed!'}]      
    return prompt

def action_selection(actions, prompt, init_prompt):
    def edit_act(action):
        if action.startswith('put'):
            action = action.replace(' in ', ' in/on ').replace(' on ', ' in/on ')
        return action.strip()
    def perform_step_ahead(action, prmpt):
            nxt_action = edit_act(llm(init_prompt + prompt + f' {action}\nOK.\n>', k=1, stop=['\n']))
            return nxt_action
    #preprocessing
    actions = [edit_act(action) for action in actions]
    think_count = [1 if action.startswith('think') else 0 for action in actions]
    think_count = sum(think_count)
    if think_count < len(actions)//2:
        if args.verbose: print(f'Generated: {actions}')
	if args.score == 'majority': action = max(set(actions), key=actions.count)
	else: action = score_select(actions, prompt, init_prompt, args.score)
    else:
        # majority of actions require thinking, perform lookahead:
        lookup = []
        next_acts = []
        act_strs = []
        for a, action in enumerate(actions):
            if action.startswith('think'):
                act_str = f' {action}\nOK.\n>'
                next_action = perform_step_ahead(action, prompt + act_str)
                tpat = 0
                while next_action.startswith('think') and tpat < 2:
                    action = next_action
                    act_str += f' {action}\nOK.\n>'
                    next_action = perform_step_ahead(action, prompt + act_str)
                    tpat += 1
                act_strs.append(act_str)
                next_acts.append(next_action)
        if args.verbose: print(f'Generated: {act_strs}')
        if args.verbose: print(f'Leads to: {next_acts}')
	if args.score == 'majority': next_action = max(set(next_acts), key=next_acts.count)
	else: next_action = score_select(next_acts, act_strs, init_prompt, args.score)
        prompt += act_strs[next_acts.index(next_action)]
        action = next_action
    if args.verbose: print('Selecting: ', action)
    return action, prompt
        

def alfworld_run(prompt, to_print=True, ob='', env=env, max_runs=max_runs, output_term=True):
    if isinstance(prompt, list): 
        init_prompt = copy.copy(prompt)
        init_prompt.append({'type': 'env', 'content': ob})
    else:
        init_prompt = prompt + '\n' + ob + '\n>'
    prompt = ''
    action_history = []
    max_patience = 5
    pat_ctr = 0
    success = False
    terminate = False
    num_runs = 0
    if to_print:
        print(ob)
        sys.stdout.flush()
    for i in range(1, max_runs):
        if args.k == 1: action = llm(init_prompt + prompt, k=args.k, stop=['\n']).strip()
        else:
            actions = llm(init_prompt + prompt, k=args.k, stop=['\n'])
            action, prompt = action_selection(actions, prompt, init_prompt)
        num_runs += 1
        action = action.lstrip('> ')
        if action.startswith('put'):
            action = action.replace(' in ', ' in/on ').replace(' on ', ' in/on ')
        observation, reward, done, info = env.step([action])
        observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]

        if action.startswith('think:'):
            observation = 'OK.'
            if 'task completed!' in action.lower(): done = True; success = True; #print(action)
            if 'task failed!' in action.lower(): done = True; success = False; #print(action)
        else: action_history.append(action)
        if observation == "Nothing happens." or observation == "OK.":
            pat_ctr += 1
            if pat_ctr == max_patience: terminate = True; break
        else: pat_ctr = 0
        if to_print:
            print(f'Act {i}: {action}\nObs {i}: {observation}')
            sys.stdout.flush()
        prompt += f' {action}\n{observation}\n>'
        if reward: success = True; terminate = False
        if done:
            return reward, success, terminate, prompt, action_history, num_runs
    if not done: success = False; terminate = True
    return 0, success, terminate,  prompt, action_history, num_runs

outputs = {k:{} for k in prefixes.keys()}
run_count = [0] * 6
cnts = [0] * 6
rs = [0] * 6
rate = 0.0
pbar = tqdm(sel_game_files)
pbar.set_postfix({'success rate': rate})
for game in pbar:
    env = orig_env.init_game(batch_size=1, game_file=game)
    ob, info = env.reset()
    ob = '\n'.join(ob[0].split('\n\n')[1:])
    name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
    room, task = ob.split('\n')
    task = task.split(': ')[-1]
    custom_ob = room + '\nYour task is to: ' + task
    trace = []
    for i, (k, v) in enumerate(prefixes.items()):
        if name.startswith(k):
            if args.executor == 'atomic':
                r, succ, term, trace, actions, num_runs = alfworld_run(atomic_exec_prompt, to_print=args.verbose, ob=custom_ob, env=env)
            elif 'hybrid' == args.executor:
                if 'davinci' in exec_LM or 'turbo-instruct' in exec_LM or 'llama' in exec_LM.lower():
                    prompt = '\n'.join(atomic_exec_prompt.split('\n')[:-1])
                elif 'gpt-3.5' in exec_LM or 'gpt-4' in exec_LM:
                    prompt = atomic_exec_prompt[:-1]
                else: prompt = '\n'.join(atomic_exec_prompt.split('\n')[:-1])
                prompt = prompt + fetch_react_prompt(k, prefixes, d, 'common')
                r, succ, term, trace, actions, num_runs = alfworld_run(prompt, to_print=args.verbose, ob=custom_ob, env=env)
            elif 'react' == args.executor:
                r, succ, term, trace, actions, num_runs = alfworld_run(fetch_react_prompt(k,prefixes,d,args.react_type), to_print=args.verbose, ob=custom_ob, env=env)
            rs[i] += r
            cnts[i] += 1
            run_count[i] += num_runs/args.num_task_samples
            rate = sum(rs)/sum(cnts)
            outputs[k][name] = {'problem': ob, 'trace': trace, 'reward': r, 'runs': num_runs}
            pbar.set_postfix({'rate': rate})
            break
    if args.verbose: print('\n\n')
print('rs', rs, 'cnts', cnts, 'sum(rs)/sum(cnts)', sum(rs) / sum(cnts), 'runs: ', sum(run_count)/6)
outputs['overall'] = {'rate': sum(rs) / sum(cnts), 'runs': sum(run_count)/6, 'success': rs, 'count': cnts}
if args.store_results:
    json.dump(outputs, t, indent=4)
    t.close()

        
