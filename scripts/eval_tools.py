# -*- coding: utf-8 -*-
import json
import argparse
import os
import re
import nltk
import numpy as np
import subprocess
import tempfile

def moses_multi_bleu(hypotheses, references, lowercase=False):
    """Calculate the bleu score using the MOSES ulti-bleu.perl script.
    Args:
    hypotheses: A numpy array of strings where each string is a single example.
    references: A numpy array of strings where each string is a single example.
    lowercase: If true, pass the "-lc" flag to the multi-bleu script
    Returns:
    The BLEU score as a float32 value.
    """

    if np.size(hypotheses) == 0:
        return np.float32(0.0)

    # Set MOSES multi-bleu script path
    metrics_dir = os.path.dirname(os.path.realpath(__file__))
    multi_bleu_path = os.path.join(metrics_dir, "multi-bleu.perl")

    # Dump hypotheses and references to tempfiles
    hypothesis_file = tempfile.NamedTemporaryFile()
    hypothesis_file.write("\n".join(hypotheses).encode("utf-8"))
    hypothesis_file.write(b"\n")
    hypothesis_file.flush()
    reference_file = tempfile.NamedTemporaryFile()
    reference_file.write("\n".join(references).encode("utf-8"))
    reference_file.write(b"\n")
    reference_file.flush()

    # Calculate BLEU using multi-bleu script
    bleu_score = 0.0
    with open(hypothesis_file.name, "r") as read_pred:
        bleu_cmd = [multi_bleu_path]
        if lowercase:
            bleu_cmd += ["-lc"]
        bleu_cmd += [reference_file.name]
        try:
            bleu_out = subprocess.check_output(bleu_cmd, stdin=read_pred, stderr=subprocess.STDOUT)
            bleu_out = bleu_out.decode("utf-8")
            bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
            bleu_score = float(bleu_score)
        except subprocess.CalledProcessError as error:
            if error.output is not None:
                print("multi-bleu.perl script returned non-zero exit code")
                print(error.output)
                bleu_score = np.float32(0.0)

    # Close temp files
    hypothesis_file.close()
    reference_file.close()

    return bleu_score

def eval_bleu(output_fp):
    """
    eval BLEU score of predicted questions
    """
    hyps = []
    refs = []
    with open(output_fp, 'r') as fr:
        for line in fr:
            sample = json.loads(line.strip())
            if isinstance(sample["pred"], list):
                pred = sample["pred"][0]
            else:
                pred = sample["pred"]
            hyps.append(pred)
            refs.append(sample["question"])
    assert len(hyps) == len(refs)
    hyp_arrys = np.array(hyps)
    ref_arrys = np.array(refs)
    bleu_score = moses_multi_bleu(hyp_arrys, ref_arrys, lowercase=True)
    return bleu_score


def get_template_match(pred_str, gold_str):
    pred = pred_str.lower()
    gold = gold_str.lower()
    template_keys = [
        "select one to refine",
        "to know about",
        "do you mean",
        "are you looking for",
        "to do with",
        "who are you shopping for",
        "what are you trying to do",
        "do you have any"
    ]
    score = 0
    for tk in template_keys:
        if tk in pred and tk in gold:
            score = 1.0
    return score

def get_exact_match(pred_str, gold_str):
    stop_words = ["a", "an", "the", "this", "that"]   # do not compare prep. words
    equal_words = ["do you want", "would you like"]   # equal representation
    pred_str = pred_str.lower()
    gold_str = gold_str.lower()

    if equal_words[1] in pred_str:
        pred_str = pred_str.replace(equal_words[1], equal_words[0])
    if equal_words[1] in gold_str:
        gold_str = gold_str.replace(equal_words[1], equal_words[0])

    pred = nltk.word_tokenize(pred_str)
    pred = [w for w in pred if not w in stop_words]
    gold = nltk.word_tokenize(gold_str)
    gold = [w for w in gold if not w in stop_words]
    if pred == gold:
        return 1
    else:
        return 0

def get_exact_match_by_key(sample):
    if sample["pred_template"] == sample["question_template"] \
        and sample["pred_slot"] == sample["question_slot"]:
        return 1
    else:
        return 0

def eval_acc(output_fp):
    """
    eval accuracy of predicted questions
    """
    count = 0
    acc_tm = 0
    acc_em = 0
    with open(output_fp, 'r') as fr:
        for line in fr:
            sample = json.loads(line.strip())
            acc_tm += get_template_match(sample["pred"][0], sample["question"])
            acc_em += get_exact_match(sample["pred"][0], sample["question"])
            #acc_em += get_exact_match_by_key(sample)
            count += 1
    acc_tm_score = float(acc_tm) / count
    acc_em_score = float(acc_em) / count
    return acc_tm_score, acc_em_score

def eval_MRR(output_fp, top_k=3):
    """
    eval MRR@k of predicted questions
    """
    count = 0
    mrr_score = 0.0
    with open(output_fp, 'r') as fr:
        for line in fr:
            sample = json.loads(line.strip())
            pred = list(sample["pred"])
            gold = sample["question"]
            score = 0.0
            for rank, pred_str in enumerate(pred[:top_k]):
                if get_template_match(pred_str, gold) == 1.0:
                    score = 1.0 / (rank + 1)
                    break
            mrr_score += score
            count += 1
    mrr_score = float(mrr_score) / count
    return mrr_score

def eval_entityF1(output_fp, slot_fp):
    """
    eval micro Entity F1 of slot entity in predicted questions
    """
    all_ents = set()
    with open(slot_fp, 'r') as fr:
        for line in fr:
            w = line.strip().split('\t')[0]
            all_ents.add(w)

    tp_count, fp_count, fn_count = 0, 0, 0
    with open(output_fp, 'r') as fr:
        for line in fr:
            sample = json.loads(line.strip())
            pred_q = sample["pred"][0] if isinstance(sample["pred"], list) else sample["pred"]
            gold_ent = sample["query"] if sample["question_slot"] == '<QUERY>' else sample["question_slot"]
            if not gold_ent in all_ents:
                all_ents.add(gold_ent)
            if gold_ent == '<EMPTY>':
                continue
            else:
                # count tp, fp, fn
                if gold_ent in pred_q:
                    tp_count += 1
                else:
                    fn_count += 1
                for w in nltk.word_tokenize(pred_q):
                    if (w in all_ents) and (w not in gold_ent):
                        fp_count += 1
    precision = tp_count / float(tp_count + fp_count) if (tp_count + fp_count) != 0 else 0
    recall = tp_count / float(tp_count + fn_count) if (tp_count + fn_count) != 0 else 0
    f1_score = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
    return f1_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", type=str)
    parser.add_argument("--eval_metric", type=str, choices=["ACC", "MRR", "BLEU", "Entity-F1"])
    parser.add_argument("--slot_file", default="data/slot_vocab.txt", type=str)
    parser.add_argument("--top_k", default=3, type=int)
    args = parser.parse_args()
    print("Evaluate [%s]" % args.eval_file)
    
    if args.eval_metric == "ACC":
        # eval Acc
        acc_tm, acc_em = eval_acc(args.eval_file)
        print("ACC_tm: %.3f" % acc_tm)
        print("ACC_em: %.3f" % acc_em)
    elif args.eval_metric == "MRR":
        # eval MRR
        mrr = eval_MRR(args.eval_file, top_k=args.top_k)
        print("MRR@%d: %.3f" % (args.top_k, mrr))
    elif args.eval_metric == "BLEU":
        # eval BLEU
        bleu = eval_bleu(args.eval_file)
        print("BLEU: %.3f" % bleu)
    elif args.eval_metric == "Entity-F1":
        # eval Entity F1
        entity_f1 = eval_entityF1(args.eval_file, args.slot_file)
        print("Entity F1: %.3f" % entity_f1)
    else:
        raise ValueError("eval_metric should be set within `ACC`, `MRR`, `BLEU`, `Entity-F1`!")
