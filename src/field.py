# -*- coding: UTF-8 -*-
import re
import os
import logging
import nltk
import torch
from tqdm import tqdm
from collections import Counter

PAD = "_pad_"
UNK = "_unk_"
BOS = "_bos_"
EOS = "_eos_"
NUM = "_num_"
SEP = "_sep_"


def tokenize(s, lower_case=True):
    """
    tokenize
    """
    s = s.replace("[SEP]", SEP)
    if lower_case:
        s = s.lower()
    s = re.sub(r'\d+.\d+', NUM, s)  # for float numbers
    s = re.sub(r'\d+', NUM, s)      # for other numbers
    tokens = nltk.word_tokenize(s, language="english")
    return tokens

class Field(object):
    """
    Field
    """
    def __init__(self,
                 sequential=False,
                 dtype=None):
        self.sequential = sequential
        self.dtype = dtype if dtype is not None else int

    def str2num(self, string):
        """
        str2num
        """
        raise NotImplementedError

    def num2str(self, number):
        """
        num2str
        """
        raise NotImplementedError

    def numericalize(self, strings):
        """
        numericalize
        """
        if isinstance(strings, str):
            return self.str2num(strings)
        else:
            return [self.numericalize(s) for s in strings]

    def denumericalize(self, numbers):
        """
        denumericalize
        """
        if isinstance(numbers, torch.Tensor):
            with torch.cuda.device_of(numbers):
                numbers = numbers.tolist()
        if self.sequential:
            if not isinstance(numbers[0], list):
                return self.num2str(numbers)
            else:
                return [self.denumericalize(x) for x in numbers]
        else:
            if not isinstance(numbers, list):
                return self.num2str(numbers)
            else:
                return [self.denumericalize(x) for x in numbers]


class TextField(Field):
    """
    TextField
    """
    def __init__(self,
                 pad_token=PAD,
                 unk_token=UNK,
                 bos_token=BOS,
                 eos_token=EOS,
                 num_token=NUM,
                 sep_token=SEP,
                 max_vocab_size=30000,
                 special_tokens=None,
                 embed_file=None,
                 embed_size=None,
                 lower_case=True):
        super(TextField, self).__init__(sequential=True,
                                        dtype=int)
        self.tokenize_fn = tokenize   # self-defined tokenize function
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.num_token = num_token
        self.sep_token = sep_token
        self.max_vocab_size = max_vocab_size
        self.embed_file = embed_file
        self.embed_size = embed_size
        self.lower_case = lower_case

        specials = [self.pad_token, self.unk_token, self.bos_token, self.eos_token,
                    self.num_token, self.sep_token]
        self.specials = [x for x in specials if x is not None]
        if special_tokens is not None:
            for token in special_tokens:
                if self.lower_case:
                    token = token.lower()
                if token not in self.specials:
                    self.specials.append(token)

        self.itos = []
        self.stoi = {}
        self.vocab_size = 0
        self.embeddings = None

    def build_vocab(self, texts, vocab_file=None, min_freq=0):
        """
        build_vocab
        """
        def flatten(xs):
            """
            flatten
            """
            flat_xs = []
            for x in xs:
                if isinstance(x, str):
                    flat_xs.append(x)
                elif isinstance(x[0], str):
                    flat_xs += x
                else:
                    flat_xs += flatten(x)
            return flat_xs
        logging.info("Building vocabulary from raw data...")
        # flatten texts
        texts = flatten(texts)

        counter = Counter()
        for string in tqdm(texts):
            tokens = self.tokenize_fn(string, lower_case=self.lower_case)
            counter.update(tokens)

        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        for tok in self.specials:
            del counter[tok]

        self.itos = list(self.specials)

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        cover = 0
        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == self.max_vocab_size:
                break
            self.itos.append(word)
            cover += freq
        cover = cover / sum(freq for _, freq in words_and_frequencies)
        logging.info(
            "Built vocabulary of size {} (coverage: {:.3f})".format(len(self.itos), cover))

        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.vocab_size = len(self.itos)

        if vocab_file is not None:
            self.dump_vocab(vocab_file=vocab_file)
        if self.embed_file is not None:
            self.embeddings = self.build_word_embeddings(self.embed_file, self.embed_size)

    def build_word_embeddings(self, embed_file, embed_dim):
        """
        build_word_embeddings
        """
        if isinstance(embed_file, list):
            embeds = [self.build_word_embeddings(e_file)
                      for e_file in embed_file]
        elif isinstance(embed_file, dict):
            embeds = {e_name: self.build_word_embeddings(e_file)
                      for e_name, e_file in embed_file.items()}
        else:
            cover = 0
            embeds = [[0] * embed_dim] * len(self.stoi)
            logging.info("Building word embeddings from '{}' ...".format(embed_file))
            with open(embed_file, "r") as f:
                for line in f:
                    w, vs = line.rstrip().split(maxsplit=1)
                    if w in self.stoi:
                        try:
                            vs = [float(x) for x in vs.split(" ")]
                        except Exception:
                            vs = []
                        if len(vs) == embed_dim:
                            embeds[self.stoi[w]] = vs
                            cover += 1
            rate = cover / len(embeds)
            logging.info("{} words have pretrained word embeddings (coverage: {:.3f})".format( \
                    cover, rate))
        return embeds

    def dump_vocab(self, vocab_file):
        """
        dump_vocab
        """
        with open(vocab_file, 'w') as fw:
            for word in self.itos:
                fw.write(word)
                fw.write('\n')

    def load_vocab(self, vocab_file):
        """
        load_vocab
        """
        with open(vocab_file, 'r') as fr:
            for line in fr:
                word = line.strip()
                self.itos.append(word)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.vocab_size = len(self.itos)
        if self.embed_file is not None:
            self.embeddings = self.build_word_embeddings(self.embed_file, self.embed_size)

    def str2num(self, string):
        """
        str2num
        """
        tokens = []
        unk_idx = self.stoi[self.unk_token]

        if self.bos_token:
            tokens.append(self.bos_token)

        tokens += self.tokenize_fn(string, lower_case=self.lower_case)

        if self.eos_token:
            tokens.append(self.eos_token)
        indices = [self.stoi.get(tok, unk_idx) for tok in tokens]
        return indices

    def num2str(self, number):
        """
        num2str
        """
        tokens = [self.itos[x] for x in number]
        if tokens[0] == self.bos_token:
            tokens = tokens[1:]
        text = []
        for w in tokens:
            if w != self.eos_token:
                text.append(w)
            else:
                break
        text = [w for w in text if w not in (self.pad_token, )]
        text = " ".join(text)
        return text
