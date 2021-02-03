from typing import List, Tuple
import torch


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

def collate_question(query, template, slot):
    """
    Collate a question based on template and slot
    """ 
    T1 = "select one to refine your search"
    T2 = "what (do you want | would you like) to know about (.+)?"
    T3 = "(which | what) (.+) do you mean?"
    T4 = "(what | which) (.+) are you looking for?"
    T5 = "what (do you want | would you like) to do with (.+)?"
    T6 = "who are you shopping for?"
    T7 = "what are you trying to do?"
    T8 = "do you have any (specific | particular) (.+) in mind?"

    question = None
    if slot == "<QUERY>":
        slot = query
    if template == T1 or template == T6 or template == T7:
        question = template
    elif template == T2:
        if slot == query:
            question = "what do you want to know about %s?" % slot
        else:
            question = "what do you want to know about this %s?" % slot
    elif template == T3:
        question = "which %s do you mean?" % slot
    elif template == T4:
        question = "what %s are you looking for?" % slot
    elif template == T5:
        question = "what do you want to do with %s?" % slot
    elif template == T8:
        question = "do you have any specific %s in mind?" % slot
    else:
        raise ValueError("Error of template!")
    return question
