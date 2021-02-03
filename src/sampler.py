import random


class NegativeSampler():
    """
    Randomly sample candidates from a list of candidates.
    Args:
        candidates: list of str containing the candidates
        num_candidates_samples: int containing the number of negative samples for each query.
    """
    def __init__(self, candidates, num_candidates_samples=0, seed=42):
        random.seed(seed)
        self.candidates = candidates
        self.num_candidates_samples = num_candidates_samples
        
    def sample(self, relevant_q):
        """
        Samples from a list of candidates randomly.
        """
        assert self.num_candidates_samples > 0
        sample_candidates = []
        for d in self.candidates:
            if not d == relevant_q:
                sample_candidates.append(d)
        sampled = random.sample(sample_candidates, self.num_candidates_samples)
        return sampled
