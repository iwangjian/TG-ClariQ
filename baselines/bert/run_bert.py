# -*- coding: utf-8 -*-
import logging
import os
import sys
import torch
import random
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers.data.processors.utils import InputFeatures
from model import TransformerRanker
from dataset import SimpleDataset, QueryDocumentDataLoader
from data_collator import DefaultDataCollator
from sampler import NegativeSampler
from utils import acumulate_list

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s [%(levelname)s] %(message)s",
  handlers=[
      logging.StreamHandler(sys.stdout)
  ]
)

def load_data(data_dir, data_partition, task):
    assert data_partition in ('train', 'dev')
    assert task in ('qs', 'ts')
    data = []
    with open("%s/MIMICS-%s.txt" % (data_dir, data_partition)) as fr:
        for line in fr:
            sample = json.loads(line)
            query = sample["query"]
            pages = " [SEP] ".join(list(sample["pages"]))
            input_text = query + " [SEP] " + pages
            if task == 'qs':
                question = sample["question"]
            else:
                question = sample["question_template"]
            data.append([input_text, question])
    data = pd.DataFrame(np.array(data))
    data.columns = ["query", "question"]
    return data

def load_candidates(data_dir, task, is_test=False):
    assert task in ('qs', 'ts')
    # sample negative samples for training using the question bank
    if task == 'qs':
        if is_test:
            cand_file = "%s/all_questions_test.txt" % data_dir
        else:
            cand_file = "%s/all_questions.txt" % data_dir
    else:
        cand_file = "%s/all_templates.txt" % data_dir
    question_bank = []
    with open(cand_file, 'r') as fr:
        for line in fr:
            question_bank.append(line.strip())
    return question_bank
        

def run_train(args):
    train = load_data(data_dir=args.data_dir, data_partition='train', task=args.task)
    dev = load_data(data_dir=args.data_dir, data_partition='dev', task=args.task)
    question_bank = load_candidates(data_dir=args.data_dir, task=args.task)
    print("train data: ", len(train))
    print("dev data: ", len(dev))
    print("all candidates: ", len(question_bank))

    # use negative samples during training
    ns_sampler = NegativeSampler(candidates=question_bank, 
        num_candidates_samples=args.ns_num, 
        task=args.task)
    
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    cache_dir = "%s/cache_data" % args.log_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    tokenizer = BertTokenizer.from_pretrained("%s/vocab.txt" % args.bert_model_dir)
    dataloader = QueryDocumentDataLoader(
        train_df=train, val_df=dev, test_df=dev,
        tokenizer=tokenizer, negative_sampler_train=ns_sampler,
        negative_sampler_val=ns_sampler, task_type='classification',
        train_batch_size=args.batch_size, val_batch_size=args.batch_size, max_seq_len=args.max_seq_len,
        sample_data=-1, cache_path=cache_dir)
    train_loader, val_loader, test_loader = dataloader.get_pytorch_dataloaders()
    
    # use BERT (any model that has SequenceClassification class from HuggingFace would work here)
    model = BertForSequenceClassification.from_pretrained(args.bert_model_dir)
    ranker = TransformerRanker(
        model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        task_type="classification", tokenizer=tokenizer,
        validate_steps=1000, show_steps=100, num_validation_instances=-1,
        num_epochs=args.num_epochs, lr=args.lr, sacred_ex=None)
    
    tokenizer.save_pretrained(args.log_dir)
    ranker.fit(log_dir=args.log_dir)


def run_test(args):
    question_bank = load_candidates(data_dir=args.data_dir, task=args.task, is_test=True)
    print("all candidates: ", len(question_bank))
    test_data = []
    examples = []
    with open("%s/MIMICS-test.txt" % args.data_dir) as fr:
        for idx, line in enumerate(fr):
            sample = json.loads(line)
            query = sample["query"]
            if args.task == 'qs':
                question = sample["question"]
            else:
                question = sample["question_template"]
            test_data.append({
                "query": query,
                "question": question,
            })
            pages = " [SEP] ".join(list(sample["pages"]))
            input_text = query + " [SEP] " + pages
            for q in question_bank:
                examples.append((input_text, q))
    print("input examples:", len(examples))
    
    prepared_path = "%s/cache_data/test_n_cand_all.pkl" % (args.log_dir)
    if os.path.exists(prepared_path):
        with open(prepared_path, 'rb') as f:
            logging.info("Loading instances from {}".format(prepared_path))
            instances = pickle.load(f)
    else:
        tokenizer = BertTokenizer.from_pretrained("%s/vocab.txt" % args.log_dir)
        logging.info("Encoding examples using tokenizer.batch_encode_plus().")
        batch_encoding = tokenizer.batch_encode_plus(examples, max_length=args.max_seq_len,
                                                truncation=True, padding='max_length')
        logging.info("Transforming examples to instances format.")
        instances = []
        for i in range(len(examples)):
            inputs = {k: batch_encoding[k][i] for k in batch_encoding}
            feature = InputFeatures(**inputs, label=0)
            instances.append(feature)
        # save tokenized instances
        with open(prepared_path, 'wb') as f:
            pickle.dump(instances, f)
        logging.info("Saved to [%s]" % prepared_path)
    
    dataset = SimpleDataset(instances)
    data_collator = DefaultDataCollator()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, 
        collate_fn=data_collator.collate_batch)
    
    # load fine-tuned model
    model = BertForSequenceClassification.from_pretrained(args.log_dir)
    ranker = TransformerRanker(
        model=model, train_loader=None, val_loader=None, test_loader=None,
        task_type="classification", tokenizer=tokenizer,
        validate_steps=1000, show_steps=100, num_validation_instances=-1,
        num_epochs=args.num_epochs, lr=args.lr, sacred_ex=None)
    _, _, softmax_output = ranker.predict(dataloader, is_test=True)
    
    softmax_output_by_query = acumulate_list(softmax_output[0], len(question_bank))
    preds = []
    for idx, sample in enumerate(test_data):
        all_scores = np.array(softmax_output_by_query[idx])
        top_idxs = (-all_scores).argsort()[:args.top_k]
        pred_question = [question_bank[topi] for topi in top_idxs]
        preds.append({
            "query": sample["query"],
            "question": sample["question"],
            "pred": pred_question
        })

    # save output
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_path = "%s/test_output_%s.txt" % (args.output_dir, args.task)
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
    parser.add_argument("--task", default='', type=str, choices=['qs', 'ts'])
    parser.add_argument("--data_dir", default='', type=str)
    parser.add_argument("--bert_model_dir", default='', type=str)
    parser.add_argument("--log_dir", default='', type=str)
    parser.add_argument("--output_dir", default='', type=str)
    parser.add_argument("--ns_num", default=2, type=int)
    parser.add_argument("--max_seq_len", default=512, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_epochs", default=5, type=int)
    parser.add_argument("--lr", default=5e-6, type=float)
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
