import os
import openai
import sys
import json
import re
from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt, # type: ignore
    wait_random_exponential, # type: ignore
    RetryError
)
import copy
import ast
import utils
import argparse
import transformers

openai.api_key = os.environ['OPENAI_KEY']

LM = 'llama'

max_depth = 1
page_len = 10

global cart 
cart = []

transformers.set_seed(4)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(15))
def llm(prompt, stop=["\n"], max_tokens=250):
    if 'davinci' in LM or 'instruct' in LM:
        response = openai.Completion.create(
        model=LM,
        prompt=prompt,
        temperature=0,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop
        )
        return response["choices"][0]["text"]

    elif 'llama' in LM.lower():
        if isinstance(prompt, list): prompt = prompt[0]
        prompt= "You are a helpful assistant navigating through a shopping website.\n" + prompt
        response = utils.opensource_completion(prompt, 250)
        return response
        
    elif 'gpt-3.5-turbo' in LM or 'gpt-4' in LM:
        api_key = os.environ.get("AZURE_API_KEY")
        openai.api_key = api_key
        openai.api_base = "https://instance-east-us2.openai.azure.com/"
        openai.api_type = "azure"
        openai.api_version = "2023-05-15"
        if 'gpt-3.5-turbo' in LM:
          deployment_name = "gpt-35-turbo" 
        elif 'gpt-4' in LM:
          deployment_name = 'gpt4'
        response = openai.ChatCompletion.create(
        engine=deployment_name,
        messages=[
          {"role": "system", "content": 'You are a helpful assistant navigating through a shopping website'},
          {"role": "user", "content": prompt}
	      ],
        temperature=0.7,
        max_tokens=max_tokens,
        top_p=0.85,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop
        )
        choices = response["choices"]
        completion_objs = [choice.message for choice in choices]
        completions = [completion.content for completion in completion_objs]
        return completions[0]

import requests
from bs4 import BeautifulSoup
from bs4.element import Comment

global WEBSHOP_URL
ACTION_TO_TEMPLATE = {
    'Description': 'description_page.html',
    'Features': 'features_page.html',
    'Reviews': 'review_page.html',
    'Attributes': 'attributes_page.html',
}

def clean_str(p):
  return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


def tag_visible(element):
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return (
        element.parent.name not in ignore and not isinstance(element, Comment)
    )


def webshop_text(session, page_type, query_string='', page_num=1, asin='', options={}, subpage='', **kwargs):
    if page_type == 'init':
      url = (
          f'{WEBSHOP_URL}/{session}'
      )
    if page_type == 'search':
      url = (
          f'{WEBSHOP_URL}/search_results/{session}/'
          f'{query_string}/{page_num}'
      )
    elif page_type == 'item':
      url = (
          f'{WEBSHOP_URL}/item_page/{session}/'
          f'{asin}/{query_string}/{page_num}/{options}'
      )
    elif page_type == 'item_sub':
      url = (
          f'{WEBSHOP_URL}/item_sub_page/{session}/'
          f'{asin}/{query_string}/{page_num}/{subpage}/{options}'
      )
    elif page_type == 'end':
      url = (
          f'{WEBSHOP_URL}/done/{session}/'
          f'{asin}/{options}'
      )
    html = requests.get(url).text
    html_obj = BeautifulSoup(html, 'html.parser')
    texts = html_obj.findAll(text=True)
    visible_texts = list(filter(tag_visible, texts))
    
    if False:
        # For `simple` mode, return just [SEP] separators
        return ' [SEP] '.join(t.strip() for t in visible_texts if t != '\n')
    else:
        # Otherwise, return an observation with tags mapped to specific, unique separators
        observation = ''
        option_type = ''
        options = {}
        asins = []
        cnt = 0
        prod_cnt = 0
        just_prod = 0
        for t in visible_texts:
            if t == '\n': continue
            if t.replace('\n', '').replace('\\n', '').replace(' ', '') == '': continue
    
            if t.parent.name == 'button':  # button
                processed_t = f'\n[{t}] '
            elif t.parent.name == 'label':  # options
                if f"'{t}'" in url:
                    processed_t = f'[[{t}]]'
                else:
                    processed_t = f'[{t}]'
                options[str(t)] = option_type
            elif t.parent.get('class') == ["product-link"]: # product asins
                processed_t = f'\n[{t}] '
                if prod_cnt >= 10:
                  processed_t = ''
                prod_cnt += 1
                asins.append(str(t))
                just_prod = 0
            else: # regular, unclickable text
                processed_t =  '\n' + str(t) + ' '
                if cnt < 2 and page_type != 'init': processed_t = ''
                if just_prod <= 2 and prod_cnt >= page_len + 1: processed_t = ''
                option_type = str(t)
                cnt += 1
            just_prod += 1
            observation += processed_t
        info = {}
        if options:
          info['option_types'] = options
        if asins:
          info['asins'] = asins
        if 'Your score (min 0.0, max 1.0)' in visible_texts:
          idx = visible_texts.index('Your score (min 0.0, max 1.0)')
          info['reward'] = float(visible_texts[idx + 1])
          observation = 'Your score (min 0.0, max 1.0): ' + (visible_texts[idx + 1])
        return clean_str(observation), info, url

class webshopEnv:
  def __init__(self):
    self.sessions = {}
    self.url_history = {}

  def clone_state(self):
    return copy.deepcopy(self.sessions)
  
  def step(self, session, action):
    done = False
    observation_ = None
    if action == 'reset':
      self.sessions[session] = {'session': session, 'page_type': 'init'}
    elif action == 'load':
      # observation_, _, _ = webshop_text(**self.sessions[session])
      self.sessions[session] = self.sessions[session]
    elif action.startswith('think['):
      observation = 'OK.'
    elif action.startswith('search['):
      assert self.sessions[session]['page_type'] == 'init'
      query = action[7:-1]
      self.sessions[session] = {'session': session, 'page_type': 'search',
                                'query_string': query, 'page_num': 1}
    elif action.startswith('click['):
      button = action[6:-1]
      if button == 'Buy Now':
        assert self.sessions[session]['page_type'] == 'item'
        self.sessions[session]['page_type'] = 'end'
        done = True
      elif button == 'Back to Search':
        assert self.sessions[session]['page_type'] in ['search', 'item_sub', 'item']
        self.sessions[session] = {'session': session, 'page_type': 'init'}
      elif button == 'Next >':
        # assert False # ad hoc page limitation
        assert self.sessions[session]['page_type'] == 'search'
        self.sessions[session]['page_num'] += 1
      elif button == '< Prev':
        assert self.sessions[session]['page_type'] in ['search', 'item_sub', 'item']
        if self.sessions[session]['page_type'] == 'search':
          # assert False
          self.sessions[session]['page_num'] -= 1
        elif self.sessions[session]['page_type'] == 'item_sub':
          self.sessions[session]['page_type'] = 'item'
        elif self.sessions[session]['page_type'] == 'item':
          self.sessions[session]['page_type'] = 'search'
          self.sessions[session]['options'] = {}
      elif button in ACTION_TO_TEMPLATE:
        assert self.sessions[session]['page_type'] == 'item'
        self.sessions[session]['page_type'] = 'item_sub'
        self.sessions[session]['subpage'] = button
      else:
        if self.sessions[session]['page_type'] == 'search':
          assert button in self.sessions[session].get('asins', [])  # must be asins
          self.sessions[session]['page_type'] = 'item'
          self.sessions[session]['asin'] = button
        elif self.sessions[session]['page_type'] == 'item':
          assert 'option_types' in self.sessions[session]
          assert button in self.sessions[session]['option_types'], (button, self.sessions[session]['option_types'])  # must be options
          option_type = self.sessions[session]['option_types'][button]
          if not 'options' in self.sessions[session]:
            self.sessions[session]['options'] = {}
          self.sessions[session]['options'][option_type] = button
          observation_ = f'You have clicked {button}.'
    else:
      assert False
    observation, info, url = webshop_text(**self.sessions[session])
    if action == 'reset': observation = observation.replace('Instruction:  \n', 'Instruction:  \nI am looking to buy a product. ')
    if observation_:
      observation = observation_
    self.sessions[session].update(info)
    # self.url_history
    reward = info.get('reward', 0.0)
    return observation, reward, done,

env = webshopEnv()

def custom_webshop_run(idx, prompt, env, to_print=False, cart=[]):
  ## Assumption is that task is contained in the prompt.
  init_prompt = prompt
  prompt = ''
  history = []
  act_history = []
  pat_ctr = 0
  max_pat = 3
  done = False
  observation = ''
  action = "load"
  for i in range(10):
    try:
      if not (action.startswith('think') or action.startswith('load') or action == 'reset' or 'cart' in action): act_history.append(action)
      res = env.step(idx, action)
      observation = res[0]
      pat_ctr = 0
      
    except AssertionError:
      observation = 'Invalid action! Try a different action.'
      pat_ctr += 1
    
    if action.startswith('cart'):
      # print('Notice Action: ', action)
      prod_id = action.replace('cart[', '').split(']')[0]
      if len(prod_id) == 10:
        cart.append(prod_id)
      observation = 'OK.'
      history.append(f'Action: {action}\nObservation: {observation}')
      break

    if action.startswith('think'):
      observation = 'OK.'

    if 'load' in action or 'reset' in action:
      observation = observation


    if to_print:
      print(f'Action: {action}\nObservation: {observation}\n')
      sys.stdout.flush()
    if i:
      prompt += f' {action}\nObservation: {observation}\n\nAction:'
    else:
      prompt += f'{observation}\n\nAction:'
  
    history.append(f'Action: {action}\nObservation: {observation}')
    # print(f'Action: {action}\nObservation: {observation}\n')
    

    if pat_ctr >= max_pat: 
      print('exhausted patience')
      break
    
    if res[2] or (action.startswith('think') and ('task completed' in action.lower() or 'task failed' in action.lower())):  #Finds the done variable, gives reward
      done = True
      return (res[0], res[1], done), history, act_history, cart

    action = llm(init_prompt + prompt[-(6400-len(init_prompt)):], stop=['\n']).lstrip(' ') #History is being shortened
  return (res[0], 0, done), history, act_history, cart 

def load_checkpoint(idx, ckpt):
    alt_env = webshopEnv()
    for act in ckpt:
        res = alt_env.step(idx, act)
        print(act)
        print(res)
        print(alt_env.sessions[idx])
        print('------')

buy_prompt="""Instruction: Buy product [B078GWRC1J] that matches criteria: 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars
You can use the following commands with the environment:
- search: To search for a query provided you are on a search page with a [search] button
- click: To click on a link enclosed within '[]'. E.g., to buy use 'click[Buy Now]'
- think: To add comments or thoughts without affecting the state of the webpage

Action: think['I should be on the search page where I can see a list of products, I should verify that.']
Observation: OK.

Action: load
Observation:
[Back to Search] 
Page 1 (Total results: 50) 
[Next >] 
[B078GWRC1J] 
Bright Citrus Deodorant by Earth Mama | Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B08KBVJ4XN] 
Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
$35.95 
[B078GTKVXY] 
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 

Action: think['I can spot product [B078GWRC1J] and it is a clickable link since its enclosed in square brackets. I should click on it.]
Observation: OK.

Action: click[B078GWRC1J]
Observation:
[Back to Search] 
[< Prev] 
scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
Price: $10.99 
Rating: N.A. 
[Description] 
[Features] 
[Reviews] 
[Buy Now]

Action: think[I'm already on the product page.  I see multiple possible options enclosed in '[]' for scent and size for this product. To match the input criteria, I should click on options '[bright citrus]' for scent and '[3 ounce (pack of 1)]' for size one by one and then buy in the end.]
Observation: OK.

Action: click[bright citrus]
Observation: You have clicked bright citrus. 

Action: click[3 ounce (pack of 1)]
Observation: You have clicked 3 ounce (pack of 1). 

Action: think[My task is to buy the product, for it should to click 'buy now']
Observation: OK.

Action: click[Buy Now]
Observation: You have clicked buy now.

Action: think[I finished buying the product. Task completed!]


Here is another task in which you need to buy a product. When you finish buying the product with the most relevant choices, use 'think[Task completed']. If you cannot find the matching options or proceed, think['Task failed']. Note that you can only click on text enclosed in '[]' on the webpage. Everything else is only a description, not valid with the "click" action.

Instruction: Buy product [{}] that matches the criteria: {}
    
"""

selection_prompt = """Instruction: Your task is to select a product that matches the user criteria and add it to your 'cart'.
You can use the following commands with the environment:
- search: To search for a query provided you are on a search page with a [search] button
- click: To click on a link enclosed within '[]'
- think: To add comments or thoughts without affecting the state of the webpage
- cart: To add an item to the cart from which it can be bought later.


Here are a few examples:
Criteria: I want to buy a 3 ounce bottle of citrus deodorant for sensitive skin and clinically tested that is priced lower than $30 and natural.

Action: search[3 ounce bright citrus deodorant sensitive skin]
Observation: 
[Back to Search] 
Page 1 (Total results: 50) 
[Next >] 
[B078GWRC1J] 
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B078GTKVXY] 
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B08KBVJ4XN] 
Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
$15.95  

Action: think[B078GWRC1J and B078GTKVXY are bright citrus deodorant less then 50 dollars. I can check B078GWRC1J first to check if it matches all my requirements.]
Observation: OK.

Action: click[B078GWRC1J]
Observation: 
[Back to Search] 
[< Prev] 
scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
Price: $10.99 
Rating: N.A. 
[Description] 
[Features] 
[Reviews] 
[Attributes]
[Buy Now]  

Action: think[I should explore the features of this product to verify it matches all my requirements.]
Observation: OK.

Action: click[Features]
Observation:
[Back to Search]
[< Prev]
 NEW from Earth Mama (formerly Earth Mama Angel Baby), formulated especially for pregnancy, breastfeeding and sensitive skin  
 Contains organic grapefruit, tangerine and calendula  
 NO propylene glycol, artificial fragrance, parabens or aluminum  
 Dermatologist tested and clinically tested for irritation 
 Better than natural organic! NSF/ANSI 305 Certified by Oregon Tilth   

Action: think[B078GWRC1J matches my requirements: natural, citrus deodrant, for sensitive skin, clinincally tested, and price less than $30. Thus, I should put it in my cart.]
Observation: OK.

Action: cart[B078GWRC1J]
Observation: OK.

Criteria: {}

"""

def search(env, idx, query):
    hist = []
    if not env.sessions[idx]['page_type'] == 'init': 
        res = env.step(idx, 'click[Back to Search]')
        hist.append('click[Back to Search]')
        # return env, ('Not on a valid page.', 0, False), {'a_hist':[]}
    res = env.step(idx, f'search[{query}]')
    hist.append(f'search[{query}]')
    return env, (res[0], res[1], True), {'a_hist': hist}

def buy(env, idx, prod_id, query):
    # pure LLM involvement --> React
    # print('Attempting buy', prod_id, ' for ', query)
    if not env.sessions[idx]['page_type'] == 'search': return env, ('Not on search page with list of items.', 0, False), {'a_hist':[]}
    obs, _, _ = webshop_text(**env.sessions[idx]) 
    prompt = buy_prompt.format(prod_id, query)
    res, l_hist, a_hist, _ = custom_webshop_run(idx, prompt, env, False, [])
    # if 'Error' in res[0]: return env, (res[0], res[1], False), {'a_hist':a_hist}
    return env, (res[0], res[1], res[2]), {'a_hist':a_hist, 'l_hist':l_hist}

def select(env, idx, criteria, cart = []):
  hist = []
  if not env.sessions[idx]['page_type'] == 'init': 
    res = env.step(idx, 'click[Bsack to Search]')
    hist.append('click[Back to Search]')
  prompt = selection_prompt.format(criteria)
  res, l_hist, a_hist, cart = custom_webshop_run(idx, prompt, env, False, cart)
  return env, (res[0], res[1], res[2]), {'a_hist': a_hist, 'l_hist': l_hist, 'cart': cart}


def webshop_run(idx, prompt, env, task, to_print=True):
  action = 'reset'
  init_prompt = prompt
  prompt = ''
  history = []
  act_history = []
  pat_ctr = 0
  max_pat = 3
  for i in range(10):
    try:
      if not (action.startswith('think') or action.startswith('load') or action == 'reset'): act_history.append(action)
      res = env.step(idx, action)
      observation = res[0]
      pat_ctr = 0
      
    except AssertionError:
      observation = 'Invalid action! Try a different action.'
      pat_ctr += 1

    if action == 'reset':
      observation = f'Your task is to  :\n{task}'

    if action.startswith('think'):
      observation = 'OK.'

    if 'load' in action:
      observation = observation


    if to_print:
      print(f'Action: {action}\nObservation: {observation}\n')
      sys.stdout.flush()
    if i:
      prompt += f' {action}\nObservation: {observation}\n\nAction:'
    else:
      prompt += f'{observation}\n\nAction:'
  
    history.append(f'Action: {action}\nObservation: {observation}')
    # print(f'Action: {action}\nObservation: {observation}\n')
    

    if pat_ctr >= max_pat: 
      print('exhausted patience')
      break
    
    if res[2] or (action.startswith('think') and ('task completed' in action.lower() or 'task failed' in action.lower())):  #Finds the done variable, gives reward
      return res[1], history, act_history, env.clone_state(), res[-1]

    action = llm(init_prompt + prompt[-(6400-len(init_prompt)):], stop=['\n']).lstrip(' ') #History is being shortened

    # print('predicted action: ', action)
  return 0, history, act_history, env.clone_state(), res[-1]

def run_episodes(n=50, kconst = 1, start_idx = 100, verbose=False):
  rs = []
  cnt = 0
  logs = {}
  act_logs = {}
  score = 0
  sr = 0
  rr = 0
  for i in range(start_idx, start_idx + n):
    print(f'Starting fixed_{i} ...')
    hist = []
    env = webshopEnv()
    res = env.step(f'fixed_{i}', 'reset')
    obs = res[0]
    task = obs.split('Instruction:  \nI am looking to buy a product. ')[-1].capitalize().split('\n')[0]
    # print(task)
    logs[f'fixed_{i}'] = {}
    logs[f'fixed_{i}']['problem'] = task.split('\n')[0]
    logs[f'fixed_{i}']['products'] = []
    logs[f'fixed_{i}']['scores'] = []
    logs[f'fixed_{i}']['reward'] = []
    pbar = tqdm(range(kconst))
    pbar.set_postfix({'score': score})
    for k in pbar:
      run_logs = {}
      try:
        cart, hist  = [], []
        _ = env.step(f'fixed_{i}', 'reset')
        env, res, meta = select(env, f'fixed_{i}', task, [])
        a_hist = meta['a_hist']
        hist = meta['l_hist']
        cart = meta['cart']
        logs[f'fixed_{i}']['products'].extend(cart)
        if not len(cart): continue
        prod_id = cart[0]
        run_logs['product'] = prod_id

        for act in a_hist:
          if act.startswith('search['):
            break
        _ = env.step(f'fixed_{i}',"click[Back to Search]")
        _ = env.step(f'fixed_{i}', act)

        a_hist.extend(["click[Back to Search]", act])
        env, res, meta = buy(env, f'fixed_{i}', prod_id, task)
        ba_hist = meta['a_hist']
        bhist = meta['l_hist']
        r = res[1]
      except AssertionError:
        r = 0
        hist, a_hist, ba_hist, bhist = [], [], [], []
        cnt += 1 
        run_logs = {}
      rs.append(r)
      run_logs['score'] = r
      run_logs['success'] = r == 1
      run_logs['act_history'] = {'select': a_hist, 'buy': ba_hist}
      run_logs['trace'] = {'select': hist, 'buy': bhist}
      logs[f'fixed_{i}']['scores'].append(r)
      logs[f'fixed_{i}']['reward'].append(r == 1)
      logs[f'fixed_{i}'][f'run_{k}'] = run_logs 
      pbar.set_postfix({'score': sum(rs) / len(rs)})
    
    
    
  score, sr, fr = sum(rs) / len(rs), len([_ for _ in rs if _ == 1]) / n, cnt / n
  logs['prompts'] = {'select': selection_prompt, 'buy': buy_prompt}
  
  return rs, logs


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--port", type=str, default='3000', help="Which port of deployed webshops should I use?")
  parser.add_argument("--k", type=int, default=20, help="Number of self consistency runs")
  parser.add_argument("--num_sess", type=int, default=10, help="Number of task instances")
  parser.add_argument("--start", type=int, default=100, help="Number of task instances")

  args = parser.parse_args()

  ## To run ReACT
  WEBSHOP_URL = "http://127.0.0.1:" + args.port
  num_sess = args.num_sess
  k = args.k
  res, logs = run_episodes(num_sess, kconst=k, start_idx=args.start, verbose=False)
  os.makedirs(f"results/webshop/CL34_{LM}/", exist_ok=True) 
  json.dump(logs, open(f'results/webshop/CL34_{LM}/seed2_react_traj{k}_pgdisp{page_len}_from{args.start}_sess{num_sess}.json', "w+"), indent=4)


