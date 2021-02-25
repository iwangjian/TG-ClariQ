# -*- coding: utf-8 -*-
import json
from rank_bm25 import BM25Okapi
import nltk
import os

def build_bm25_corpus(f_paths):
    """
    Reads files and build bm25 corpus (index)
    """
    question_corpus = []
    question_set = set()
    for fp in f_paths:
        with open(fp, 'r') as fr:
            for line in fr:
                sample = json.loads(line)
                if not sample["question"] in question_set:
                    question_set.add(sample["question"])
                    question_toks = [word for word in nltk.word_tokenize(sample["question"])]
                    question_corpus.append(question_toks)
    assert len(question_corpus) == len(question_set)
    print("Building bm25 corpus: {}".format(len(question_corpus)))
    return question_corpus

def run_bm25(question_corpus, test_path, out_dir, task, top_k=3):
    """
    Run bm25 for every query and store output
    """
    preds = []
    bm25 = BM25Okapi(question_corpus)
    print("Runing bm25 retrieving...")
    with open(test_path, 'r') as fr:
        for line in fr:
            sample = json.loads(line)
            q = [word for word in nltk.word_tokenize(sample["query"].lower())]
            pages = " ".join(sample["pages"]).lower()
            q += [word for word in nltk.word_tokenize(pages)]
            bm25_list = bm25.get_top_n(q, question_corpus, n=top_k)
            # select top-k ranked output
            pred_question = [" ".join(bm25_list[idx]) for idx in range(top_k)]
            preds.append({
                "query": sample["query"],
                "question": sample["question"],
                "question_template": sample["question_template"],
                "question_slot": sample["question_slot"],
                "pred": pred_question
            })
    out_path = "%s/test_output_%s.txt" % (out_dir, task)
    with open(out_path, 'w') as fw:
        for s in preds:
            line = json.dumps(s)
            fw.write(line)
            fw.write('\n')
    print("Saved to [%s]" % out_path)


if __name__ == "__main__":
    train_fp = "data/MIMICS-train.txt"
    dev_fp = "data/MIMICS-dev.txt"
    test_fp = "data/MIMICS-test.txt"
    cand_fp = "data/all_templates.txt"
    out_dir = "output/bm25"
    top_k = 3

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Task qs: question selection
    print("Task: question selection")
    bm25_corpus = build_bm25_corpus([train_fp, dev_fp, test_fp])
    run_bm25(bm25_corpus, test_fp, out_dir, task='qs', top_k=top_k)

    print("\nTask: template selection")
    question_bank = []
    with open(cand_fp, 'r') as fr:
        for line in fr:
            question_toks = [word for word in nltk.word_tokenize(line.strip())]
            question_bank.append(question_toks)
    run_bm25(question_bank, test_fp, out_dir, task='ts', top_k=top_k)
    