#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os

DATA_DIR = "data"
CACHE_DIR = "log/seq2seq/cache_data"
MODES = ["train", "dev", "test"]


def main():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    for mode in MODES:
        src_data = []
        tgt_data = []
        with open("%s/MIMICS-%s.txt"% (DATA_DIR, mode)) as fr:
            for line in fr:
                sample = json.loads(line)
                query = sample["query"]
                pages = " [SEP] ".join(list(sample["pages"]))
                src_text = query + " [SEP] " + pages
                tgt_text = sample["question"]
                src_data.append(src_text)
                tgt_data.append(tgt_text)
        print("{}: src-{} tgt-{}".format(mode.upper(), len(src_data), len(tgt_data)))

        with open("%s/src-%s.txt" % (CACHE_DIR, mode), 'w') as fw:
            for src in src_data:
                fw.write(src)
                fw.write('\n')
        print("Saved to [%s/src-%s.txt]" % (CACHE_DIR, mode))
        with open("%s/tgt-%s.txt" % (CACHE_DIR, mode), 'w') as fw:
            for tgt in tgt_data:
                fw.write(tgt)
                fw.write('\n')
        print("Saved to [%s/tgt-%s.txt]" % (CACHE_DIR, mode))

if __name__ == "__main__":
    main()
    