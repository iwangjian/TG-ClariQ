import numpy as np
import random
import os
import logging
import pickle


class NegativeSampler():
    """
    Randomly sample candidates from a list of candidates.
    Args:
        candidates: list of str containing the candidates
        num_candidates_samples: int containing the number of negative samples for each query.
    """
    def __init__(self, candidates, num_candidates_samples, task, seed=42):
        random.seed(seed)
        self.candidates = candidates
        self.num_candidates_samples = num_candidates_samples
        self.task = task

        self.template_keys = [
            "select one to refine",
            "to know about",
            "do you mean",
            "are you looking for",
            "to do with",
            "who are you shopping for",
            "what are you trying to do",
            "do you have any"
        ]
        self.question_dicts = {}
        for k in self.template_keys:
            self.question_dicts[k] = []
        for cand_q in self.candidates:
            for k in self.template_keys:
                if k in cand_q.lower():
                    self.question_dicts[k].append(cand_q)

    def random_sample(self, relevant_docs):
        """
        Samples from a list of candidates randomly.
        """
        assert len(relevant_docs) == 1
        relevant_doc = relevant_docs[0]

        sample_candidates = []
        for d in self.candidates:
            if not d == relevant_doc:
                sample_candidates.append(d)
        sampled = random.sample(sample_candidates, self.num_candidates_samples)
        return sampled
    
    def hierachi_sample(self, relevant_docs):
        """
        Samples from a list of candidates by different categories.
        """
        assert len(relevant_docs) == 1
        relevant_doc = relevant_docs[0]
        
        hier_sampled = []
        for k in self.template_keys:
            if k in relevant_doc.lower():
                continue
            else:
                sq = random.sample(self.question_dicts[k], 1)[0]
                hier_sampled.append(sq)
        sampled = random.sample(hier_sampled, self.num_candidates_samples)
        return sampled
    
    def sample(self, relevant_docs):
        if self.task == 'ts':
            return self.random_sample(relevant_docs)
        else:
            return self.hierachi_sample(relevant_docs)
