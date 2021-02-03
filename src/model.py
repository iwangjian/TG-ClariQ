# -*- coding: utf-8 -*-
import logging
import torch
import torch.nn as nn
from transformers import BertModel
from embedder import Embedder
from rnn_encoder import RNNEncoder
from modules import AttentionLayer, FeedForwardLayer


class NeurClariQuestion(nn.Module):
   
    def __init__(self, 
                encoder_name, 
                hidden_size,
                slot_size,
                num_labels,
                vocab_size=30000,
                embed_size=300,
                padding_idx=None,
                bert_config=None,
                num_attention_heads=1, 
                num_layers=1,
                dropout_prob=0.1, 
                lambda1=1.0, 
                lambda2=1.0):
        super(NeurClariQuestion, self).__init__()
        assert encoder_name in ('gru', 'lstm', 'bert')
        self.encoder_name = encoder_name
        self.hidden_size = hidden_size
        self.slot_size = slot_size
        self.num_labels = num_labels
        self.num_attention_heads = num_attention_heads
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        if self.encoder_name == 'gru' or self.encoder_name == 'lstm':
            self.embedder = Embedder(num_embeddings=vocab_size,
                                     embedding_dim=embed_size,
                                     padding_idx=padding_idx)
            self.query_encoder = RNNEncoder(input_size=embed_size,
                                            hidden_size=self.hidden_size,
                                            name=self.encoder_name,
                                            embedder=self.embedder,
                                            num_layers=1,
                                            bidirectional=False,
                                            dropout=self.dropout_prob)
            self.question_encoder = RNNEncoder(input_size=embed_size,
                                               hidden_size=self.hidden_size,
                                               name=self.encoder_name,
                                               embedder=self.embedder,
                                               num_layers=1,
                                               bidirectional=False,
                                               dropout=self.dropout_prob)
        else:
            # load pretrained Bert model just for encoder
            self.query_encoder = BertModel.from_pretrained(bert_config)
            #for p in self.query_encoder.parameters():
            #    p.requires_grad = False
            self.question_encoder = BertModel.from_pretrained(bert_config)
            #for p in self.question_encoder.parameters():
            #    p.requires_grad = False
        
        self.query_self_attention = nn.ModuleList([AttentionLayer(hidden_size=self.hidden_size, 
                                                   num_attention_heads=self.num_attention_heads,
                                                   dropout_prob=self.dropout_prob) 
                                                   for _ in range(self.num_layers)])
        self.question_self_attention = nn.ModuleList([AttentionLayer(hidden_size=self.hidden_size, 
                                                      num_attention_heads=self.num_attention_heads,
                                                      dropout_prob=self.dropout_prob)
                                                      for _ in range(self.num_layers)])
        self.cross_attention = nn.ModuleList([AttentionLayer(hidden_size=self.hidden_size,
                                            num_attention_heads=self.num_attention_heads,
                                            dropout_prob=self.dropout_prob)
                                            for _ in range(self.num_layers)])
        self.cross_ffn = nn.ModuleList([FeedForwardLayer(hidden_size=self.hidden_size) 
                                        for _ in range(self.num_layers)])
        self.query_ffn = nn.ModuleList([FeedForwardLayer(hidden_size=self.hidden_size)
                                        for _ in range(self.num_layers)])
        
        self.scoring_layer = nn.Linear(self.hidden_size * 2, self.num_labels)
        self.sloting_layer = nn.Linear(self.hidden_size * 2, self.slot_size)
        self.scoring_loss = nn.CrossEntropyLoss(reduction='sum')
        self.sloting_loss = nn.CrossEntropyLoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info("Device {}".format(self.device))
        self.to(self.device)
        

    def forward(self, input_ids_query, input_ids_question,
                attention_mask_query, attention_mask_question,
                label_rank=None, label_slot=None):
        if self.encoder_name == 'gru' or self.encoder_name == 'lstm':
            query_hidden = self.query_encoder(inputs=input_ids_query)
            question_hidden = self.question_encoder(inputs=input_ids_question)
        else:
            query_hidden = self.query_encoder(input_ids=input_ids_query, 
                                              attention_mask=attention_mask_query)
            question_hidden = self.question_encoder(input_ids=input_ids_question,
                                                attention_mask=attention_mask_question)
        
        query_outputs, cross_outputs = query_hidden[0], question_hidden[0]
        for layer in range(self.num_layers):
            # calcaulate query
            query_attn_outputs = self.query_self_attention[layer](hidden_states=query_outputs,
                                                    attention_mask=attention_mask_query)
            query_outputs = self.query_ffn[layer](query_attn_outputs[0])

            # calculate question
            question_attn_outputs = self.question_self_attention[layer](hidden_states=cross_outputs,
                                                attention_mask=attention_mask_question)

            # calculate cross-attention between query and question
            cross_attn_outputs = self.cross_attention[layer](hidden_states=question_attn_outputs[0],
                                                encoder_hidden_states=query_attn_outputs[0],
                                                output_attentions=True)
            cross_outputs = self.cross_ffn[layer](cross_attn_outputs[0])

        # concat different representations into shared representation
        pooler_query = torch.mean(query_outputs, dim=1, keepdim=False)   # shape(batch_size, hidden_size)
        pooler_cross = torch.mean(cross_outputs, dim=1, keepdim=False)   # shape(batch_size, hidden_size)
        shared_repr = torch.cat([pooler_query, pooler_cross], dim=1)

        # calculate logits for different tasks
        scoring_logits = self.scoring_layer(shared_repr)
        sloting_logits = self.sloting_layer(shared_repr)

        if label_rank is not None and label_slot is not None:
            scoring_loss = self.scoring_loss(input=scoring_logits.view(-1, self.num_labels), 
                                             target=label_rank.view(-1))
            
            sloting_loss = self.sloting_loss(input=sloting_logits.view(-1, self.slot_size), 
                                             target=label_slot.view(-1))
            loss = self.lambda1 * scoring_loss + self.lambda2 * sloting_loss
        else:
            loss = None
        outputs = (loss, scoring_logits, sloting_logits)
        return outputs
