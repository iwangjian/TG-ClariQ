# -*- coding: utf-8 -*-
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers.optimization import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
from evaluation import evaluate, evaluate_and_aggregate
from utils import acumulate_list_multiple_relevant, acumulate_l1_by_l2


class Trainer(object):
    def __init__(self, model, train_loader, dev_loader, log_dir,
                log_steps, validate_steps, num_epochs, lr, weight_decay=0.01, max_grad_norm=0.5):
        
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.log_dir = log_dir
        self.log_steps = log_steps
        self.validate_steps = validate_steps
        self.num_epochs = num_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm

        if self.lr is not None:
            #self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer, 
                                            num_warmup_steps=5000, num_training_steps=96000)
        self.best_metric = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def train(self):
        """
        Train the transformer-based neural model
        """
        logging.info("Total batches per epoch : {}".format(len(self.train_loader)))
        logging.info("Validating every {} batches.".format(self.validate_steps))        
        val_metric = 0
        for epoch in range(self.num_epochs):
            logging.info("\nEpoch {}:".format(epoch + 1))
            for batch_step, inputs in enumerate(self.train_loader):
                self.model.train()
                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)                

                outputs = self.model(**inputs)
                loss = outputs[0]

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                if batch_step > 0 and batch_step % self.log_steps == 0:
                    logging.info("batch: {}\tloss: {:.3f}".format(batch_step, loss))
                if batch_step > 0 and batch_step % self.validate_steps == 0:
                    predicts = self.predict(loader=self.dev_loader)
                    res = evaluate_and_aggregate(predicts["all_ranking_logits"], 
                                                predicts["all_ranking_labels"], 
                                                ['map'])
                    rank_map = res["map"]
                    slot_acc = evaluate(predicts["all_sloting_softmax_logits"],
                                        predicts["all_sloting_labels"])
                    val_metric = rank_map + slot_acc
                    if val_metric > self.best_metric:
                        self.best_metric = val_metric
                        logging.info("Best validation -- MAP: {:.3f} ACC: {:.3f}".format(rank_map, slot_acc))
                        model_to_save = "%s/best_model.bin" % self.log_dir
                        torch.save(self.model, model_to_save)
                        logging.info("Saved to [%s/best_model.bin]" % self.log_dir)
                    else:
                        logging.info("Validation -- MAP: {:.3f} ACC: {:.3f}".format(rank_map, slot_acc))              
            logging.info("Epoch {} training done.".format(epoch + 1))
            model_to_save = "%s/model_epoch_%d.bin" % (self.log_dir, epoch + 1)
            torch.save(self.model, model_to_save)
            logging.info("Saved to [%s/model_epoch_%d.bin]" % (self.log_dir, epoch + 1))
            

    def predict(self, loader, is_test=False):
        """
        Uses trained model to make predictions on the loader.
        Args:
            loader: the DataLoader containing the set to run the prediction and evaluation.         
        Returns:
            Matrices (logits, labels, softmax_logits)
        """
        self.model.eval()
        all_ranking_labels, all_sloting_labels = [], []
        all_ranking_logits, all_sloting_logits = [], []
        all_ranking_softmax_logits, all_sloting_softmax_logits = [], []
        if is_test:
            for batch in tqdm(loader, total=len(loader)):
                for k, v in batch.items():
                    batch[k] = v.to(self.device)
                with torch.no_grad():
                    outputs = self.model(**batch)
                    ranking_logits, sloting_logits = outputs[1], outputs[2]
                    all_ranking_labels += batch["label_rank"].tolist()
                    all_ranking_logits += ranking_logits[:, 1].tolist()
                    all_ranking_softmax_logits += F.softmax(ranking_logits, dim=1)[:, 1].tolist()

                    all_sloting_labels += batch["label_slot"].tolist()
                    all_sloting_logits += sloting_logits.tolist()
                    all_sloting_softmax_logits += F.softmax(sloting_logits, dim=1).tolist()
        else:
            # validation
            for idx, batch in enumerate(loader):
                for k, v in batch.items():
                    batch[k] = v.to(self.device)
                with torch.no_grad():
                    outputs = self.model(**batch)
                    ranking_logits, sloting_logits = outputs[1], outputs[2]
                    all_ranking_labels += batch["label_rank"].tolist()
                    all_ranking_logits += ranking_logits[:, 1].tolist()
                    all_ranking_softmax_logits += F.softmax(ranking_logits, dim=1)[:, 1].tolist()
                    
                    all_sloting_labels += batch["label_slot"].tolist()
                    all_sloting_logits += sloting_logits.tolist()
                    all_sloting_softmax_logits += F.softmax(sloting_logits, dim=1).tolist()
            # accumulate per query for ranking task
            all_ranking_labels = acumulate_list_multiple_relevant(all_ranking_labels)
            all_ranking_logits = acumulate_l1_by_l2(all_ranking_logits, all_ranking_labels)
            all_ranking_softmax_logits = acumulate_l1_by_l2(all_ranking_softmax_logits, all_ranking_labels)
        
        outputs = {
            "all_ranking_logits": all_ranking_logits,
            "all_sloting_logits": all_sloting_logits,
            "all_ranking_softmax_logits": all_ranking_softmax_logits,
            "all_sloting_softmax_logits": all_sloting_softmax_logits,
            "all_ranking_labels": all_ranking_labels,
            "all_sloting_labels": all_sloting_labels,
        }
        return outputs

