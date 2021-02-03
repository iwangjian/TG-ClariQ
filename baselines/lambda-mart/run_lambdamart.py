# -*- coding: utf-8 -*-
import logging
import os
import sys
import json
import argparse
import subprocess
import numpy as np
from sampler import NegativeSampler
from dataset import RankDataset

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
            data.append((input_text, question))
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
    logging.info("All candidates: {}".format(len(question_bank)))

    # use negative samples during training
    ns_sampler = NegativeSampler(candidates=question_bank, 
        num_candidates_samples=args.ns_num, 
        task=args.task)
    
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    cache_dir = "%s/cache_data" % args.log_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    rank_set = RankDataset(negative_sampler=ns_sampler, max_seq_len=args.max_seq_len, 
                          base_dir=args.bert_model_dir, cache_dir=cache_dir,
                          batch_size=args.batch_size)
    train_path = rank_set.cache_features(train, data_partition='train')
    dev_path = rank_set.cache_features(dev, data_partition='dev')

    # call external Java lib
    jar_dir = os.path.dirname(os.path.realpath(__file__))
    jar_path = os.path.join(jar_dir, "RankLib-2.15.jar")
    save_path = "%s/lambdamart.model" % args.log_dir
    # -ranker <type>	Specify which ranking algorithm to use
 	#	0: MART (gradient boosted regression tree)
 	#	1: RankNet
 	#	2: RankBoost
 	#	3: AdaRank
 	#	4: Coordinate Ascent
 	#	6: LambdaMART
 	#	7: ListNet
 	#	8: Random Forests
    java_cmd = "java -jar {} -train {} -validate {} -ranker 6 -metric2t MAP -save {} -tree {} -leaf {} -shrinkage {} -tc -1 -estop {}".format(
        jar_path, train_path, dev_path, save_path, args.num_trees, args.num_leaves, args.lr, args.estop)
    return_code = subprocess.call(java_cmd, shell=True)
    print(return_code)


def run_test(args):
    question_bank = load_candidates(data_dir=args.data_dir, task=args.task, is_test=True)
    logging.info("All candidates: {}".format(len(question_bank)))
    test_data = []
    raw_test = []
    with open("%s/MIMICS-test.txt" % args.data_dir) as fr:
        for line in fr:
            sample = json.loads(line)
            query = sample["query"]
            if args.task == 'qs':
                question = sample["question"]
            else:
                question = sample["question_template"]
            raw_test.append({
                "query": query,
                "question": question,
            })
            pages = " [SEP] ".join(list(sample["pages"]))
            input_text = query + " [SEP] " + pages
            test_data.append((input_text, ))
    logging.info("Test: {}".format(len(test_data)))
    
    ns_sampler = NegativeSampler(candidates=question_bank, 
        num_candidates_samples=args.ns_num, 
        task=args.task)
    rank_set = RankDataset(negative_sampler=ns_sampler, max_seq_len=args.max_seq_len, 
                          base_dir=args.bert_model_dir, cache_dir="%s/cache_data" % args.log_dir,
                          batch_size=args.batch_size)
    test_path = rank_set.cache_features(test_data, data_partition='test')
    
    # call external Java lib
    jar_dir = os.path.dirname(os.path.realpath(__file__))
    jar_path = os.path.join(jar_dir, "RankLib-2.15.jar")
    model_path = "%s/lambdamart.model" % args.log_dir
    score_path = "%s/rank.score.txt" % args.log_dir

    java_cmd = "java -jar {} -load {} -rank {} -score {}".format(
        jar_path, model_path, test_path, score_path)
    try:
        return_code = subprocess.call(java_cmd, shell=True)
        logging.info(return_code)

        output_by_query = []
        current_l = []
        acum_step = len(question_bank)
        with open(score_path) as fr:
            for idx, line in enumerate(fr):
                score = float(line.strip().split('\t')[-1])
                current_l.append(score)
                if (idx + 1) % acum_step == 0 and idx != 0:
                    output_by_query.append(current_l)
                    current_l = []
        assert len(output_by_query) == len(raw_test)
        preds = []
        for idx, sample in enumerate(raw_test):
            all_scores = np.array(output_by_query[idx])
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
    except Exception:
        raise RuntimeError("Runtime error!")


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
    parser.add_argument("--batch_size", default=24, type=int)
    parser.add_argument("--num_trees", default=100, type=int)
    parser.add_argument("--num_leaves", default=8, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--estop", default=20, type=int)
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
