import argparse

import torch
import jsonlines
import os
import pickle

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from typing import List
from sklearn.metrics import f1_score, precision_score, recall_score

import random
import numpy as np

from tqdm import tqdm
from paragraph_model import MultiTaskSentenceClassifier as SentenceClassifier
from dataset import SciCiteDataset, SciResDataset

import logging

def reset_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def predict(model, dataset):
    model.eval()
    predictions = []
    aux1_predictions = []
    aux2_predictions = []

    with torch.no_grad():
        for batch in tqdm(DataLoader(dataset, batch_size = args.batch_size, shuffle=False)):
            encoded_dict = encode(tokenizer, batch, has_local_attention="led-" in args.repfile or "longformer" in args.repfile)
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            pred_out,_ = model(encoded_dict)
            predictions.extend([dataset.label_lookup[index] for index in pred_out])
            aux1_predictions.extend([dataset.first_label_lookup[index] for index in aux1_out])
            aux2_predictions.extend([dataset.second_label_lookup[index] for index in aux2_out])
    return predictions, aux1_predictions, aux2_predictions

def evaluation_metric(flatten_labels, flatten_predictions, mapping):
    discourse_f1 = f1_score(flatten_labels,flatten_predictions, average='micro')
    discourse_precision = precision_score(flatten_labels,flatten_predictions, average='micro')
    discourse_recall = recall_score(flatten_labels,flatten_predictions, average='micro')
    return (discourse_f1, discourse_recall, discourse_precision)

def evaluation(model, dataset):
    model.eval()
    predictions = []
    labels = []
    aux1_predictions = []
    aux1_labels = []
    aux2_predictions = []
    aux2_labels = []
        
    with torch.no_grad():
        for batch in tqdm(DataLoader(dataset, batch_size = args.batch_size, shuffle=False)):
            encoded_dict = encode(tokenizer, batch, has_local_attention="led-" in args.repfile or "longformer" in args.repfile)
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            pred_out, aux1_out, aux2_out, _, _, _= \
                    model(encoded_dict, 
                          label = batch["label"].to(device),
                          aux_label1 = batch["first_label"].to(device),
                          aux_label2 = batch["second_label"].to(device)
                         )
            predictions.extend([dataset.label_lookup[index] for index in pred_out])
            labels.extend([dataset.label_lookup[index] for index in batch["label"].tolist()])
            aux1_predictions.extend([dataset.first_label_lookup[index] for index in aux1_out])
            aux1_labels.extend([dataset.first_label_lookup[index] for index in batch["first_label"].tolist()])
            aux2_predictions.extend([dataset.second_label_lookup[index] for index in aux2_out])
            aux2_labels.extend([dataset.second_label_lookup[index] for index in batch["second_label"].tolist()])
    
    return evaluation_metric(labels, predictions, dataset.label_types), evaluation_metric(aux1_labels, aux1_predictions, dataset.first_label_types), evaluation_metric(aux2_labels, aux2_predictions, dataset.second_label_types)



def encode(tokenizer, batch, has_local_attention = False):
    inputs = batch["sentence"]
    encoded_dict = tokenizer.batch_encode_plus(
        inputs,
        padding=True, truncation=True,add_special_tokens=True,
        return_tensors='pt')
    if has_local_attention:
        #additional_special_tokens_lookup = {token: idx for token, idx in zip(tokenizer.additional_special_tokens, tokenizer.additional_special_tokens_ids)}
        #special_token_ids = set([additional_special_tokens_lookup[token] for token in special_tokens])
        #special_token_ids.add(tokenizer.mask_token_id)
        special_token_ids = tokenizer.additional_special_tokens_ids
        special_token_ids.append(tokenizer.sep_token_id)
        
        batch_size, MAX_SENT_LEN = encoded_dict["input_ids"].shape
        global_attention_mask = batch_size * [
            [0 for _ in range(MAX_SENT_LEN)] 
        ]
        for i_batch in range(batch_size):
            for i_token in range(MAX_SENT_LEN):
                if encoded_dict["input_ids"][i_batch][i_token] in special_token_ids:
                    global_attention_mask[i_batch][i_token] = 1
        encoded_dict["global_attention_mask"] = torch.tensor(global_attention_mask)
    # Single pass to BERT should not exceed max_sent_len anymore, because it's handled in dataset.py
    return encoded_dict


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train, cross-validate and run sentence sequence tagger")
    argparser.add_argument('--repfile', type=str, default = "allenai/scibert_scivocab_uncased", help="Word embedding file")
    argparser.add_argument('--train_file', type=str)
    argparser.add_argument('--pre_trained_model', type=str)
    argparser.add_argument('--test_file', type=str)
    argparser.add_argument('--bert_lr', type=float, default=1e-5, help="Learning rate for BERT-like LM")
    argparser.add_argument('--lr', type=float, default=5e-6, help="Learning rate")
    argparser.add_argument('--dropout', type=float, default=0, help="embedding_dropout rate")
    #argparser.add_argument('--bert_dim', type=int, default=768, help="bert_dimension")
    argparser.add_argument('--epoch', type=int, default=15, help="Training epoch")
    argparser.add_argument('--MAX_SENT_LEN', type=int, default=512)
    argparser.add_argument('--checkpoint', type=str, default = "multi-task_sentence_classifier.model")
    argparser.add_argument('--log_file', type=str, default = "multi-task_sentence_classifier_performances.jsonl")
    argparser.add_argument('--update_step', type=int, default=1)
    argparser.add_argument('--batch_size', type=int, default=1) # roberta-large: 2; bert: 8 
    argparser.add_argument('--aux1_coef', type=float, default=1) # 1.5
    argparser.add_argument('--aux2_coef', type=float, default=1) # 2.5  
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    
    reset_random_seed(12345)

    args = argparser.parse_args()
    #device = "cpu" ###############################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.repfile)
    #additional_special_tokens = {'additional_special_tokens': ['[BOS]']} 
    #tokenizer.add_special_tokens(additional_special_tokens)
    
    if args.train_file:
        train = True
        #assert args.repfile is not None, "Word embedding file required for training."
    else:
        train = False
    if args.test_file:
        test = True
    else:
        test = False

    params = vars(args)

    for k,v in params.items():
        print(k,v)
    
    if train:
        #train_set = SciCiteDataset(args.train_file, tokenizer, MAX_SENT_LEN = args.MAX_SENT_LEN)
        train_set = SciResDataset(args.train_file, tokenizer, MAX_SENT_LEN = args.MAX_SENT_LEN)
        
    # dev_set = SciCiteDataset(args.test_file, tokenizer, MAX_SENT_LEN = args.MAX_SENT_LEN)
    dev_set = SciResDataset(args.test_file, tokenizer, MAX_SENT_LEN = args.MAX_SENT_LEN)
    
    model = SentenceClassifier(args.repfile, len(tokenizer),args.dropout, label_size = len(dev_set.label_types), aux_label1_size = len(dev_set.first_label_types), aux_label2_size = len(dev_set.second_label_types))#.to(device)

    if args.pre_trained_model is not None:
        model.load_state_dict(torch.load(args.pre_trained_model))
    
    model = model.to(device)
    
    if train:
        settings = [{'params': model.bert.parameters(), 'lr': args.bert_lr}]
        for module in model.extra_modules:
            settings.append({'params': module.parameters(), 'lr': args.lr})
        optimizer = torch.optim.Adam(settings)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 0, args.epoch)
        model.train()

        prev_performance = 0
        for epoch in range(args.epoch):
            tq = tqdm(DataLoader(train_set, batch_size = args.batch_size, shuffle=True))
            for i, batch in enumerate(tq):
                encoded_dict = encode(tokenizer, batch, has_local_attention="led-" in args.repfile or "longformer" in args.repfile)
                encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
                pred_out, aux1_out, aux2_out, main_loss, aux1_loss, aux2_loss = \
                    model(encoded_dict,
                          label = batch["label"].to(device),
                          aux_label1 = batch["first_label"].to(device),
                          aux_label2 = batch["second_label"].to(device)
                         )
                loss = main_loss + aux1_loss * args.aux1_coef + aux2_loss * args.aux2_coef
                loss.backward()

                if i % args.update_step == args.update_step - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                    tq.set_description(f'Epoch {epoch}, iter {i}, loss: {round(loss.item(), 4)}')
            scheduler.step()

            # Evaluation
            train_score, train_aux1_score, train_aux2_score = evaluation(model, train_set)
            print(f'Epoch {epoch}, train f1 p r: %.4f, %.4f, %.4f' % train_score)
            print(f'Epoch {epoch}, train aux1 f1 p r: %.4f, %.4f, %.4f' % train_aux1_score)
            print(f'Epoch {epoch}, train aux2 f1 p r: %.4f, %.4f, %.4f' % train_aux2_score)
            
            dev_score, dev_aux1_score, dev_aux2_score = evaluation(model, dev_set)
            print(f'Epoch {epoch}, dev f1 p r: %.4f, %.4f, %.4f' % dev_score)
            print(f'Epoch {epoch}, dev aux1 f1 p r: %.4f, %.4f, %.4f' % dev_aux1_score)
            print(f'Epoch {epoch}, dev aux2 f1 p r: %.4f, %.4f, %.4f' % dev_aux2_score)
               
            dev_perf = dev_score[0]
            if dev_perf >= prev_performance:
                torch.save(model.state_dict(), args.checkpoint)
                best_state_dict = model.state_dict()
                prev_performance = dev_perf
                best_scores = (dev_score, dev_aux1_score, dev_aux2_score)
                print("New model saved!")
            else:
                print("Skip saving model.")
        
        #torch.save(model.state_dict(), args.checkpoint)
        params["f1"] = best_scores[0][0]
        params["precision"] = best_scores[0][1]
        params["recall"] = best_scores[0][2]
        
        params["aux1_f1"] = best_scores[1][0]
        params["aux1_precision"] = best_scores[1][1]
        params["aux1_recall"] = best_scores[1][2]
        
        params["aux2_f1"] = best_scores[2][0]
        params["aux2_precision"] = best_scores[2][1]
        params["aux2_recall"] = best_scores[2][2]

        with jsonlines.open(args.log_file, mode='a') as writer:
            writer.write(params)


    if test:
        if train:
            del model
            model = SentenceClassifier(args.repfile, len(tokenizer),args.dropout, label_size = len(dev_set.label_types), aux_label1_size = len(dev_set.first_label_types), aux_label2_size = len(dev_set.second_label_types)).to(device)
            model.load_state_dict(best_state_dict)
            print("Testing on the new model.")
        
        # Evaluation
        dev_score, dev_aux1_score, dev_aux2_score = evaluation(model, dev_set)
        print(f'Epoch {epoch}, test f1 p r: %.4f, %.4f, %.4f' % dev_score)
        print(f'Epoch {epoch}, test aux1 f1 p r: %.4f, %.4f, %.4f' % dev_aux1_score)
        print(f'Epoch {epoch}, test aux2 f1 p r: %.4f, %.4f, %.4f' % dev_aux2_score)
        
        params["f1"] = dev_score[0]
        params["precision"] = dev_score[1]
        params["recall"] = dev_score[2]
        
        params["aux1_f1"] = dev_aux1_score[0]
        params["aux1_precision"] = dev_aux1_score[1]
        params["aux1_recall"] = dev_aux1_score[2]
        
        params["aux2_f1"] = dev_aux2_score[0]
        params["aux2_precision"] = dev_aux2_score[1]
        params["aux2_recall"] = dev_aux2_score[2]

        with jsonlines.open(args.log_file, mode='a') as writer:
            writer.write(params)
