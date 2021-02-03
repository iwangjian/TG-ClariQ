# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import logging
import numpy as np


class Embedder(nn.Embedding):
    """
    Embedder
    """
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super(Embedder, self).__init__(num_embeddings=num_embeddings,
                                       embedding_dim=embedding_dim,
                                       padding_idx=padding_idx)

    def load_embeddings(self, embeds, scale=0.05):
        """
        load_embeddings
        """
        assert len(embeds) == self.num_embeddings
        embeds = torch.from_numpy(np.array(embeds))
        num_known = 0
        for i in range(len(embeds)):
            if len(embeds[i].nonzero()) == 0:
                nn.init.uniform_(embeds[i], -scale, scale)
            else:
                num_known += 1
        self.weight.data.copy_(embeds)
        logging.info("{} words have pretrained embeddings (coverage: {:.3f})".format(num_known, 
            float(num_known) / self.num_embeddings))
