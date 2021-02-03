import logging
import random
import os
import pickle
import pandas as pd
import dataclasses
import json
from dataclasses import dataclass
from typing import List, Optional, Union
import torch.utils.data as data
from tqdm import tqdm


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
    """

    input_ids_query: List[int]
    input_ids_question: List[int]

    attention_mask_query: Optional[List[int]] = None
    attention_mask_question: Optional[List[int]] = None

    label_rank: Optional[Union[int, float]] = None
    label_slot: Optional[Union[int, float]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


class QueryDataset(data.Dataset):
    def __init__(self, data, slot_dict, tokenizer, data_partition, cache_path,
                negative_sampler, max_seq_len=512, max_q_len=20, random_seed=42):
        random.seed(random_seed)

        self.data = data
        self.slot_dict = slot_dict
        self.tokenizer = tokenizer   # type: TextField or BertTokenizer
        self.data_partition = data_partition
        assert self.data_partition in ("train", "dev", "test")
        self.cache_path = cache_path
        self.negative_sampler = negative_sampler
        self.max_seq_len = max_seq_len
        self.max_q_len = max_q_len
        self.instances = []

        if str(self.tokenizer.__class__.__name__) == "BertTokenizer":
            self._cache_instances_bert()
        else:
            self._cache_instances()

    def _cache_instances_bert(self):
        """
        Loads tensors into memory or creates the dataset when it does not exist already.
        """        
        signature = "{}_n_cand_{}_{}.pkl".\
            format(self.data_partition,
                   self.negative_sampler.num_candidates_samples,
                   self.tokenizer.__class__.__name__)
        path = self.cache_path + "/" + signature

        if os.path.exists(path):
            with open(path, 'rb') as f:
                logging.info("Loading instances from {}".format(path))
                self.instances = pickle.load(f)
        else:            
            logging.info("Creating instances with signature {}".format(signature))

            # Creating labels (currently there is support only for binary relevance)
            relevant_label = 1
            not_relevant_label = 0
            
            examples =[]
            if self.data_partition == "test":
                for row in tqdm(self.data, total=len(self.data)):
                    query = row[0]
                    for template in self.negative_sampler.candidates:
                        examples.append((query, template, 0, 0))
            else:
                for row in tqdm(self.data, total=len(self.data)):
                    query = row[0]
                    template = row[1]
                    slot = row[2]
                    slot_label = self.slot_dict[slot]
                    examples.append((query, template, relevant_label, slot_label))
                    ns_templates = self.negative_sampler.sample(template)
                    for ns in ns_templates:
                        examples.append((query, ns, not_relevant_label, slot_label))
            examples_df = pd.DataFrame(examples)
            examples_df.columns = ['query', 'question_template', 'relevant_label', 'slot_label']
            
            logging.info("Encoding query examples using tokenizer.batch_encode_plus().")
            query_encoding = self.tokenizer.batch_encode_plus(list(examples_df['query'].values), 
                max_length=self.max_seq_len, padding='max_length', truncation=True)
            
            logging.info("Encoding question template examples using tokenizer.batch_encode_plus().")
            question_encoding = self.tokenizer.batch_encode_plus(list(examples_df['question_template'].values), 
                max_length=self.max_q_len, padding='max_length', truncation=True)

            logging.info("Transforming examples to instances format.")
            for i in range(len(examples)):
                query_inputs = {k: query_encoding[k][i] for k in query_encoding}
                question_inputs = {k: question_encoding[k][i] for k in question_encoding}
                inputs = {
                    "input_ids_query": query_inputs['input_ids'],
                    "input_ids_question": question_inputs['input_ids'],
                    "attention_mask_query": query_inputs['attention_mask'],
                    "attention_mask_question": question_inputs['attention_mask'],
                    "label_rank": examples[i][-2],
                    "label_slot": examples[i][-1]
                }
                feature = InputFeatures(**inputs)
                self.instances.append(feature)            

            with open(path, 'wb') as f:
                pickle.dump(self.instances, f)

        logging.info("Total of {} instances were cached.".format(len(self.instances)))
    

    def _batch_encode(self, str_list, max_length):
        input_ids = []
        attention_mask = []
        for text in str_list:
            text_ids = self.tokenizer.numericalize(text)
            text_masks = [1] * len(text_ids)
            if len(text_ids) > max_length:
                input_id = text_ids[:max_length]
                input_mask = text_masks[:max_length]
            else:
                pad_idx = self.tokenizer.stoi.get(self.tokenizer.pad_token)
                input_id = text_ids + [pad_idx] * (max_length-len(text_ids))
                input_mask = text_masks + [0] * (max_length-len(text_ids))
            input_ids.append(input_id)
            attention_mask.append(input_mask)
        outputs = {
            'input_ids': input_ids, 
            'attention_mask': attention_mask
        }
        return outputs

    def _cache_instances(self):
        """
        Loads tensors into memory or creates the dataset when it does not exist already.
        """        
        signature = "{}_n_cand_{}_{}.pkl".\
            format(self.data_partition,
                   self.negative_sampler.num_candidates_samples,
                   self.tokenizer.__class__.__name__)
        path = self.cache_path + "/" + signature

        if os.path.exists(path):
            with open(path, 'rb') as f:
                logging.info("Loading instances from {}".format(path))
                self.instances = pickle.load(f)
        else:            
            logging.info("Creating instances with signature {}".format(signature))

            # Creating labels (currently there is support only for binary relevance)
            relevant_label = 1
            not_relevant_label = 0
            
            examples =[]
            if self.data_partition == "test":
                for row in tqdm(self.data, total=len(self.data)):
                    query = row[0]
                    for template in self.negative_sampler.candidates:
                        examples.append((query, template, 0, 0))
            else:
                for row in tqdm(self.data, total=len(self.data)):
                    query = row[0]
                    template = row[1]
                    slot = row[2]
                    slot_label = self.slot_dict[slot]
                    examples.append((query, template, relevant_label, slot_label))
                    ns_templates = self.negative_sampler.sample(template)
                    for ns in ns_templates:
                        examples.append((query, ns, not_relevant_label, slot_label))
            examples_df = pd.DataFrame(examples)
            examples_df.columns = ['query', 'question_template', 'relevant_label', 'slot_label']
            
            logging.info("Encoding query examples using TextField.")
            query_encoding = self._batch_encode(list(examples_df['query'].values), 
                                                max_length=self.max_seq_len)
            
            
            logging.info("Encoding question template examples using TextField.")
            question_encoding = self._batch_encode(list(examples_df['question_template'].values),
                                                    max_length=self.max_q_len)

            logging.info("Transforming examples to instances format.")
            for i in range(len(examples)):
                query_inputs = {k: query_encoding[k][i] for k in query_encoding}
                question_inputs = {k: question_encoding[k][i] for k in question_encoding}
                inputs = {
                    "input_ids_query": query_inputs['input_ids'],
                    "input_ids_question": question_inputs['input_ids'],
                    "attention_mask_query": query_inputs['attention_mask'],
                    "attention_mask_question": question_inputs['attention_mask'],
                    "label_rank": examples[i][-2],
                    "label_slot": examples[i][-1]
                }
                feature = InputFeatures(**inputs)
                self.instances.append(feature)            

            with open(path, 'wb') as f:
                pickle.dump(self.instances, f)

        logging.info("Total of {} instances were cached.".format(len(self.instances)))

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        return self.instances[index]
