import numpy as np
from typing import List, Dict
from collections import Counter
from scipy import integrate, stats

class StoppingCriterias:

    def __init__(self, *args, **kwargs):

        ...

    def should_stop(self, *args, **kwargs) -> Dict:
        ...


class BetaStoppingCriteria(StoppingCriterias):

    def __init__(self, conf_thresh : float = 0.8) -> None:
        super().__init__()
        self.conf_thresh = conf_thresh

    def should_stop(self, answers : List, conf_thresh : int = None, verbose : bool = False) -> Dict:
        
        if conf_thresh is None: conf_thresh = self.conf_thresh

        
        most_common = Counter(answers).most_common(2)
        if len(most_common) == 1:
            a, b = most_common[0][1], 0
        else:
            a, b= most_common[0][1], most_common[1][1]
        a = float(a)
        b = float(b)

        return_dict = {
            'most_common' : most_common[0][0],
            'prob' : -1,
            'stop' : False,
        }
            

        try:
            prob =  integrate.quad(lambda x : x**(a) * (1-x)**(b), 0.5, 1)[0] / integrate.quad(lambda x : x**(a) * (1-x)**(b), 0, 1)[0]
            # print(prob)
        except Exception as e:
            # print error message
            print(f"Error during numerical integration: {e}")
            return_dict['stop'] = False
            return_dict['prob'] = -1
            return return_dict
        return_dict['prob'] = prob
        return_dict['stop'] = prob >= conf_thresh
        return return_dict
    
def adptive_sampling(items:List, verbose=False, threshold=None):
    if threshold is None:
        sc = BetaStoppingCriteria()
    else: sc = BetaStoppingCriteria(conf_thresh=float(threshold))
    final_items =  []
    for item in items:
        final_items.append(item)
        if sc.should_stop(final_items)['stop']:
            break
    if verbose: print('Stats: ', len(final_items)/len(items))
    if verbose: 
        if len(final_items) > 1: print('Look, non-greedy!!')
    return final_items
