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

            examples = []
            labels = []
            qids = []
            rel_score = 1
            non_rel_score = 0
            for idx, row in enumerate(tqdm(data, total=len(data))):
                qid = qid_base + idx
                query = row[0]
                if data_partition == 'test':
                    for cand in self.negative_sampler.candidates:
                        examples.append((cand, query))
                        labels.append(non_rel_score)
                        qids.append(qid)
                else:
                    # relevant
                    question = row[1]
                    examples.append((question, query))
                    labels.append(rel_score)
                    qids.append(qid)
                    # non-relevant
                    ns_cands = self.negative_sampler.sample(question)
                    for ns in ns_cands:
                        examples.append((ns, query))
                        labels.append(non_rel_score)
                        qids.append(qid)
            assert len(labels) == len(examples) and len(labels) == len(qids)
            logging.info("Examples: {}".format(len(examples)))
            logging.info("Encoding examples using tokenizer.batch_encode_plus().")
            instances = self.tokenizer.batch_encode_plus(examples, 
                            max_length=self.max_seq_len, padding='max_length', 
                            truncation=True)
            logging.info("Creating features by Bert model...")
            all_features = []
            all_hiddens = []
            i = 0
            with torch.no_grad():
                while (i + self.batch_size <= len(examples)):
                    inputs = {k: torch.tensor(instances[k][i:i+self.batch_size], dtype=torch.long) for k in instances}
                    for k, v in inputs.items():
                        inputs[k] = v.to(self.device)
                    outputs = self.model(**inputs)
                    hiddens = outputs[1].tolist()
                    all_hiddens += hiddens
                    i += self.batch_size
                if i < len(examples) and i + self.batch_size > len(examples):
                    inputs = {k: torch.tensor(instances[k][i:], dtype=torch.long) for k in instances}
                    for k, v in inputs.items():
                        inputs[k] = v.to(self.device)
                    outputs = self.model(**inputs)
                    hiddens = outputs[1].tolist()
                    all_hiddens += hiddens

            assert len(labels) == len(all_hiddens)
            for idx in range(len(all_hiddens)):
                feature = ["{}:{:.5f}".format(i+1, h) for i, h in enumerate(all_hiddens[idx])] 
                line = "{} qid:{} ".format(labels[idx], qids[idx]) + " ".join(feature)
                all_features.append(line)
            
            with open(cache_path, 'w') as fw:
                for line in all_features:
                    fw.write(line)
                    fw.write('\n')
            logging.info("Total of {} instances were cached to [{}]".format(len(all_features), cache_path))
        return cache_path
