# -*- coding: utf-8 -*-
import logging
import os
import sys
import torch
import random
import json
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from field import TextField
from dataset import QueryDataset
from data_collator import DefaultDataCollator
from sampler import NegativeSampler
from model import NeurClariQuestion
from trainer import Trainer
from utils import acumulate_list, collate_question

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s [%(levelname)s] %(message)s",
  handlers=[
      logging.StreamHandler(sys.stdout)
  ]
)

def load_data(data_dir, data_partition):
    data = []
    with open("%s/MIMICS-%s.txt" % (data_dir, data_partition)) as fr:
        for line in fr:
            sample = json.loads(line)
            query = sample["query"]
            pages = " [SEP] ".join(list(sample["pages"]))
            query_text = query + " [SEP] " + pages
            if data_partition == "test":
                question_template = None
                question_slot = None
            else:
                question_template = sample["question_template"]
                question_slot = sample["question_slot"]
            data.append((query_text, question_template, question_slot))
    template_bank = []
    with open("%s/all_templates.txt" % data_dir) as fr:
        for line in fr:
            template_bank.append(line.strip())
    slot_to_idx = {}
    idx_to_slot = {}
    with open("%s/slot_vocab.txt" % data_dir) as fr:
        for idx, line in enumerate(fr):
            w = line.strip().split('\t')[0]
            slot_to_idx[w] = idx
            idx_to_slot[idx] = w
    return data, template_bank, slot_to_idx, idx_to_slot


def run_train(args):
    train_data, template_bank, slot_to_idx, _ = load_data(data_dir=args.data_dir, data_partition="train")
    dev_data, _, _, _ = load_data(data_dir=args.data_dir, data_partition="dev")

    # use negative samples during training
    ns_sampler = NegativeSampler(template_bank, num_candidates_samples=args.ns_num)
    
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    cache_dir = "%s/cache_data" % args.log_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # use TextField tokenizer or BertTokenizer
    if args.text_encoder == "gru" or args.text_encoder == "lstm":
        tokenizer = TextField(max_vocab_size=args.max_vocab_size,
                              embed_file=args.embed_file,
                              embed_size=args.embed_size)
        vocab_file = os.path.join(cache_dir, "vocab.txt")
        if os.path.exists(vocab_file):
            tokenizer.load_vocab(vocab_file=vocab_file)
        else:
            raw_data = train_data + dev_data
            tokenizer.build_vocab(texts=raw_data, vocab_file=vocab_file)
    else:
        tokenizer = BertTokenizer.from_pretrained("%s/vocab.txt" % args.bert_model_dir)
    
    data_collator = DefaultDataCollator()
    train_dataset = QueryDataset(data=train_data, slot_dict=slot_to_idx, tokenizer=tokenizer,
        data_partition='train', cache_path=cache_dir, negative_sampler=ns_sampler,
        max_seq_len=args.max_seq_len, max_q_len=args.max_q_len
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
        collate_fn=data_collator.collate_batch
    )
    
    dev_dataset = QueryDataset(data=dev_data, slot_dict=slot_to_idx,tokenizer=tokenizer,
        data_partition='dev', cache_path=cache_dir, negative_sampler=ns_sampler,
        max_seq_len=args.max_seq_len, max_q_len=args.max_q_len
    )
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, 
        collate_fn=data_collator.collate_batch
    )

    # model definition
    if args.text_encoder == "gru" or args.text_encoder == "lstm":
        model_clariq = NeurClariQuestion(encoder_name=args.text_encoder,
            hidden_size=args.hidden_size, slot_size=len(slot_to_idx), 
            num_labels=2, vocab_size=tokenizer.vocab_size, embed_size=args.embed_size,
            padding_idx=tokenizer.stoi.get(tokenizer.pad_token),
            num_attention_heads=args.num_attention_heads, num_layers=args.num_layers
        )
        if args.embed_file is not None:
            model_clariq.embedder.load_embeddings(embeds=tokenizer.embeddings)
    else:
        model_clariq = NeurClariQuestion(encoder_name='bert',
            hidden_size=args.hidden_size, slot_size=len(slot_to_idx), 
            num_labels=2, bert_config=args.bert_model_dir,
            num_attention_heads=args.num_attention_heads, num_layers=args.num_layers
        )

    # training
    trainer = Trainer(model=model_clariq, train_loader=train_loader, dev_loader=dev_loader,
        log_dir=args.log_dir, log_steps=args.log_steps, validate_steps=args.validate_steps, 
        num_epochs=args.num_epochs, lr=args.lr
    )
    trainer.train()


def run_test(args):
    test_data, template_bank, slot_to_idx, idx_to_slot = load_data(data_dir=args.data_dir, data_partition="test")
    
    # use all negative samples during test
    ns_sampler = NegativeSampler(template_bank)
    
    # use TextField tokenizer or BertTokenizer
    if args.text_encoder == "gru" or args.text_encoder == "lstm":
        tokenizer = TextField(max_vocab_size=args.max_vocab_size,
                              embed_file=None,
                              embed_size=None)
        vocab_file = "%s/cache_data/vocab.txt" % args.log_dir
        tokenizer.load_vocab(vocab_file=vocab_file)
    else:
        tokenizer = BertTokenizer.from_pretrained("%s/vocab.txt" % args.bert_model_dir)
    
    data_collator = DefaultDataCollator()
    test_dataset = QueryDataset(data=test_data, slot_dict=slot_to_idx, tokenizer=tokenizer,
        data_partition="test", cache_path="%s/cache_data" % args.log_dir, negative_sampler=ns_sampler,
        max_seq_len=args.max_seq_len, max_q_len=args.max_q_len
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
        collate_fn=data_collator.collate_batch
    )

    # load model
    model_to_load = "%s/best_model.bin" % args.log_dir
    logging.info("Loading model from [%s]" % model_to_load)
    model_clariq = torch.load(model_to_load)

    # testing
    trainer = Trainer(model=model_clariq, train_loader=None, dev_loader=None,
        log_dir=args.log_dir, log_steps=None, validate_steps=None, 
        num_epochs=None, lr=None)
    outputs = trainer.predict(test_loader, is_test=True)

    print("all_ranking_softmax_logits:", len(outputs["all_ranking_softmax_logits"]))
    softmax_ranking_by_query = acumulate_list(outputs["all_ranking_softmax_logits"], len(template_bank))
    softmax_sloting_by_query = acumulate_list(outputs["all_sloting_softmax_logits"], len(template_bank))

    raw_test = []
    with open("%s/MIMICS-test.txt" % args.data_dir) as fr:
        for line in fr:
            sample = json.loads(line)
            raw_test.append(sample)
    assert len(raw_test) == len(test_data)
    preds = []
    for idx, sample in enumerate(raw_test):
        all_scores = np.array(softmax_ranking_by_query[idx])
        top_idxs = (-all_scores).argsort()[:args.top_k]
        # save top-k predicted templates
        pred_template = [template_bank[topi] for topi in top_idxs]
        # only save top-1 slot
        slot_idx = np.argmax(softmax_sloting_by_query[idx][top_idxs[0]])
        pred_slot = idx_to_slot[slot_idx]
        
        pred_question = [collate_question(sample["query"], pred_template[j], pred_slot) \
                            for j in range(len(pred_template))]
        preds.append({
            "query": sample["query"],
            "question": sample["question"],
            "question_template": sample["question_template"],
            "question_slot": sample["question_slot"],
            "pred_template": pred_template,
            "pred_slot": pred_slot,
            "pred": pred_question
        })

    # save output
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_path = "%s/test_output.txt" % args.output_dir
    with open(output_path, 'w') as fw:
        for s in preds:
            line = json.dumps(s)
            fw.write(line)
            fw.write('\n')
    logging.info("Saved output to [%s]" % output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--data_dir", default='', type=str)
    parser.add_argument("--log_dir", default='', type=str)
    parser.add_argument("--output_dir", default='', type=str)

    parser.add_argument("--text_encoder", default='gru', type=str, choices=['gru', 'lstm', 'bert'])
    parser.add_argument("--bert_model_dir", default='', type=str)
    parser.add_argument("--max_vocab_size", default=30000, type=int)
    parser.add_argument("--embed_file", default=None, type=str)
    parser.add_argument("--embed_size", default=300, type=int)
    parser.add_argument("--ns_num", default=7, type=int)
    parser.add_argument("--max_seq_len", default=512, type=int)
    parser.add_argument("--max_q_len", default=20, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--num_attention_heads", default=8, type=int)
    parser.add_argument("--num_layers", default=3, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_epochs", default=5, type=int)
    parser.add_argument("--log_steps", default=100, type=int)
    parser.add_argument("--validate_steps", default=1000, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--top_k", default=3, type=int)
    args = parser.parse_args()

    if args.do_train:
        run_train(args)
    elif args.do_test:
        run_test(args)
    else:
        raise ValueError("do_train or do_test should be set!")
    

if __name__ == "__main__":
    main()
