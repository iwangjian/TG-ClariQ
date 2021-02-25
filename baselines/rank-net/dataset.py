import logging
import random
import os
import pickle
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel


class RankDataset():
    def __init__(self, negative_sampler, max_seq_len, base_dir, cache_dir, batch_size=24):
        self.negative_sampler = negative_sampler
        self.max_seq_len = max_seq_len
        self.cache_dir = cache_dir
        self.tokenizer = BertTokenizer.from_pretrained("%s/vocab.txt" % base_dir)
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BertModel.from_pretrained(base_dir).to(self.device)
        self.model.eval()
    
    def _batch_encode(self, instances):
        total_num = 0
        for k, v in instances.items():
            total_num = len(v)
            break
        all_hiddens = []
        i = 0
        with torch.no_grad():
            while (i + self.batch_size <= total_num):
                inputs = {k: torch.tensor(instances[k][i:i+self.batch_size], dtype=torch.long) for k in instances}
                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)
                outputs = self.model(**inputs)
                hiddens = outputs[1].tolist()
                all_hiddens += hiddens
                i += self.batch_size
            if i < total_num and i + self.batch_size > total_num:
                inputs = {k: torch.tensor(instances[k][i:], dtype=torch.long) for k in instances}
                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)
                outputs = self.model(**inputs)
                hiddens = outputs[1].tolist()
                all_hiddens += hiddens
        return all_hiddens

    def cache_features(self, data, data_partition):
        """
        Create datasets (i.e., features) for ranklib
        """
        if data_partition == 'test':
            signature = "{}_n_cand_all.txt".format(data_partition)
        else:
            signature = "{}_n_cand_{}.txt".format(data_partition,
                   self.negative_sampler.num_candidates_samples)
        cache_path = self.cache_dir + "/" + signature

        if os.path.exists(cache_path):
            logging.info("Datasets have been cached to [{}]".format(cache_path))
        else:            
            if data_partition == 'train':
                qid_base = 100000
            elif data_partition == 'dev':
                qid_base = 10000
            else:
                qid_base = 20000

            texts_a = []
            texts_b = []
            labels = []
            qids = []
            rel_score = 1
            non_rel_score = 0
            for idx, row in enumerate(tqdm(data, total=len(data))):
                qid = qid_base + idx
                query = row[0]
                texts_a.append(query)
                if data_partition == 'test':
                    question = row[1]
                    texts_b.append(question)
                    labels.append(rel_score)
                    qids.append(qid)
                    for cand in self.negative_sampler.candidates:
                        if not cand == question:
                            texts_b.append(cand)
                            labels.append(non_rel_score)
                            qids.append(qid)
                else:
                    # relevant
                    question = row[1]
                    texts_b.append(question)
                    labels.append(rel_score)
                    qids.append(qid)
                    # non-relevant
                    ns_cands = self.negative_sampler.sample(question)
                    for ns in ns_cands:
                        texts_b.append(ns)
                        labels.append(non_rel_score)
                        qids.append(qid)
            print("texts_a:", len(texts_a))
            print("texts_b:", len(texts_b))
            assert len(labels) == len(texts_b) and len(labels) == len(qids)
            logging.info("Examples: {}".format(len(texts_b)))

            logging.info("Encoding examples using tokenizer.batch_encode_plus().")
            instances_a = self.tokenizer.batch_encode_plus(texts_a, 
                            max_length=self.max_seq_len, padding='max_length', 
                            truncation=True)
            instances_b = self.tokenizer.batch_encode_plus(texts_b,
                            max_length=20, padding='max_length',
                            truncation=True)

            logging.info("Creating features by Bert model...")
            hiddens_a = self._batch_encode(instances_a)
            hiddens_b = self._batch_encode(instances_b)
            print("hiddens_a:", len(hiddens_a))
            print("hiddens_b:", len(hiddens_b))
            all_hiddens = []
            if data_partition == 'test':
                num_interval = len(self.negative_sampler.candidates)
            else:
                num_interval = self.negative_sampler.num_candidates_samples + 1
            for idx_a, h_a in enumerate(hiddens_a):
                idx_b = idx_a * num_interval
                for h_b in hiddens_b[idx_b: idx_b+num_interval]:
                    h = h_a + h_b
                    all_hiddens.append(h)
            print("all_hiddens:", len(all_hiddens))
            print("all_hiddens(dim):", len(all_hiddens[0]))

            all_features = []
            for idx in range(len(all_hiddens)):
                feature = ["{}:{:.4f}".format(i+1, h) for i, h in enumerate(all_hiddens[idx])] 
                line = "{} qid:{} ".format(labels[idx], qids[idx]) + " ".join(feature)
                all_features.append(line)
            
            with open(cache_path, 'w') as fw:
                for line in all_features:
                    fw.write(line)
                    fw.write('\n')
            logging.info("Total of {} instances were cached to [{}]".format(len(all_features), cache_path))
        return cache_path
