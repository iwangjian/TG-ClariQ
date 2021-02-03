#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json

DATA_DIR = "data"
OUTPUT_DIR = "output/seq2seq"

def main():
    preds = []
    with open("%s/pred.txt" % OUTPUT_DIR) as fr:
        for line in fr:
            preds.append(line.strip())
    outputs = []
    with open("%s/MIMICS-test.txt" % DATA_DIR) as fr:
        for idx, line in enumerate(fr):
            sample = json.loads(line)
            out = {
                "query": sample["query"],
                "question": sample["question"],
                "question_template": sample["question_template"],
                "question_slot": sample["question_slot"],
                "pred": preds[idx]
            }
            outputs.append(out)
    assert len(outputs) == len(preds)

    with open("%s/test_output.txt" % OUTPUT_DIR, 'w') as fw:
        for s in outputs:
            line = json.dumps(s)
            fw.write(line)
            fw.write('\n')
    print("Saved to [%s/test_output.txt]" % OUTPUT_DIR)


if __name__ == "__main__":
    main()