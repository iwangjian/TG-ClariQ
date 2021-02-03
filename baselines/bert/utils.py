from typing import List, Tuple
import torch


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_input_mask, all_segment_ids, \
        all_input_pattern, all_label_ids, all_input_lens = map(torch.stack, zip(*batch))
    max_len = max(all_input_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_input_mask = all_input_mask[:, :max_len]
    all_segment_ids = all_segment_ids[:, :max_len]
    return all_input_ids, all_input_mask, all_segment_ids, all_input_pattern, all_label_ids


def acumulate_list(l : List[float], acum_step: int) -> List[List[float]]:
    """
    Splits a list every acum_step and generates a resulting matrix
    Args:
        l: List of floats
    Returns: List of list of floats divided every acum_step
    """
    acum_l = []    
    current_l = []    
    for i in range(len(l)):
        current_l.append(l[i])        
        if (i + 1) % acum_step == 0 and i != 0:
            acum_l.append(current_l)
            current_l = []
    return acum_l


def acumulate_list_multiple_relevant(l : List[float]) -> List[List[float]]:
    """
    Splits a list that has variable number of labels 1 first followed by N 0.
    Args:
        l: List of floats
    Returns: List of list of floats divided every set of 1s followed by 0s.
    Example: [1,1,1,0,0,1,0,0,1,1,0,0] --> [[1,1,1,0,0], [1,0,0], [1,1,0,0]]
    """    
    acum_l = []
    current_l = []
    for i in range(len(l)):
        current_l.append(l[i])
        if (i == len(l)-1) or (l[i] == 0 and l[i+1] == 1):
            acum_l.append(current_l)
            current_l = []
    return acum_l

def acumulate_l1_by_l2(l1 : List[float], l2 : List[List[float]]) -> List[List[float]]:
    """
    Splits a list (l1) using the shape of l2.
    Args:
        l: List of floats
    Returns: List of list of floats divided by the shape of l2
    Example: [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5] , [[1,1,1,0,0], [1,0,0], [1,1,0,0]]--> 
             [[0.5,0.5,0.5,0.5,0.5], [0.5,0.5,0.5], [0.5,0.5,0.5,0.5]]
    """    
    acum_l1 = []
    l1_idx = 0
    for l in l2:
        current_l = []
        for i in range(len(l)):
            current_l.append(l1[l1_idx])
            l1_idx+=1
        acum_l1.append(current_l)
    return acum_l1


def from_df_to_list_without_nans(df) -> List[float]:
    res = []
    for r in df.itertuples(index=False):
        l = []
        for c in range(df.shape[1]):
            if str(r[c]) != 'nan':
                l.append(r[c])
        res.append(l)
    return res
