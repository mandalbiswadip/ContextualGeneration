import argparse

import torch
import jsonlines
import json
import os
import pickle

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM, get_cosine_schedule_with_warmup

from tqdm import tqdm

import logging


def normalize_section(section):
    # Treat title and abstract as false-positive sections
    section = section.lower()
    mapping = {
        "title": "other",
        "abstract": "other",
        "introduction": "introduction",
        "conclusion": "conclusion",
        "related": "related work",
        "background": "background",
        "dataset": "dataset",
        "method": "methods",
        "algorithm": "methods",
        "approach": "methods",
        "model": "methods",
        "result": "results",
        "performance": "results",
        "model": "methods",
        "system": "methods",
        "architecture": "methods",
        "training": "methods",
        "tuning": "methods",
        "framework": "methods",
        "experiment": "results",
        "discussion": "discussion",
        "baseline": "baseline",
        "hyperparameter": "non-essential",
        "submission": "non-essential",
        "task": "task",
        "corpus": "dataset",
        "corpora": "dataset",
        "evaluat": "evaluation",
        "data": "dataset",
        "feature": "methods",
        "analysis": "results",
        "setup": "methods",
        "learning": "methods",
        "interference": "results",
        "overview": "introduction",
        "motivation": "introduction",
        "future": "discussion",
        "formulation": "methods",
        "previous": "related work",
        "limitation": "discussion",
        "acknowledgement": "non-essential",
        "implementation": "non-essential",
        "detail": "non-essential",
        "preliminaries": "background",
        "preliminary": "background",
        "example": "non-essential"
    }
    for k,v in mapping.items():
        if k in section:
            return v
    return "other"

class ParagraphDataset(Dataset):
    """
    Dataset for reading paper's each paragraph. 
    """
    def __init__(self, dataset: str, max_importance):
        importance_rank = {
            "basic": 0,
            "title": 0,
            "abstract": 1,
            "introduction": 2,
            "conclusion": 2,
            "task": 3,
            "methods": 3,
            "results": 4,
            "discussion": 4,
            "dataset": 4,
            "other": 5,
            "background": 6,
            "baseline": 6,
            "evaluation":6,
            "related work":6,
            "non-essential":7
        }
        self.samples = []
        with open(dataset,"r") as f:
            for line in tqdm(f):
                pdf_dict = json.loads(line)                
                self.samples.append({
                    "paper_id": pdf_dict["paper_id"],
                    "section": "Title",
                    "text": pdf_dict["title"],
                    "importance": importance_rank["title"]
                })
                
                for paragraph in pdf_dict["abstract"]:
                    self.samples.append({
                        "paper_id": pdf_dict["paper_id"],
                        "section": "Abstract",
                        "text": paragraph["text"],
                        "importance": importance_rank["abstract"]
                    })
                for paragraph in pdf_dict['body_text']:
                    importance = importance_rank[normalize_section(paragraph["section"])]
                    if importance <= max_importance:
                        self.samples.append({
                            "paper_id": pdf_dict["paper_id"],
                            "section": paragraph["section"],
                            "text": paragraph["text"],
                            "importance": importance
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train, cross-validate and run sentence sequence tagger")
    argparser.add_argument('--repfile', type=str, default = "allenai/scibert_scivocab_uncased", help="Word embedding file")
    argparser.add_argument('--corpus_file', type=str, default="/home/data/XiangciLi/20200705v1/acl/selected_cited_pdf_parses.jsonl")
    argparser.add_argument('--max_sent_len', type=int, default=512)
    argparser.add_argument('--batch_size', type=int, default=1) # roberta-large: 2; bert: 8
    argparser.add_argument('--max_importance', type=int, default=2)
    argparser.add_argument('--output_path', type=str, default="/home/data/XiangciLi/20200705v1/acl/embeddings")
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    
    args = argparser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.repfile)
    #model = AutoModel.from_pretrained(args.repfile).to(device)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.repfile).to(device)
    ds = ParagraphDataset(args.corpus_file, args.max_importance)
    
    prev_id = ""
    with torch.no_grad():
        tq = tqdm(DataLoader(ds, batch_size = args.batch_size, shuffle=False))
        for batch_i, batch in enumerate(tq):
            encoded_dict = tokenizer(
                batch["section"],batch["text"],
                padding='max_length',add_special_tokens=True,
                max_length=args.max_sent_len, truncation = "only_second",
                pad_to_max_length=True,
                return_tensors='pt')
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            #output = model(**encoded_dict)
            #pooler_output = output.last_hidden_state.cpu()
            output = model.get_encoder()(**encoded_dict)
            pooler_output = output.last_hidden_state.cpu()

            importance = batch["importance"]
            paper_ids = batch["paper_id"]
            sections = batch["section"]
            prev_end = 0
            current_end = 0
            for i, (paper_id, section) in enumerate(zip(batch["paper_id"], batch["section"])):
                if paper_id != prev_id:
                    if prev_id:
                        current_end = i
                        embeddings.append(pooler_output[prev_end:current_end])
                        importances.append(importance[prev_end:current_end])
                        paper_id_list.append(paper_ids[prev_end:current_end])
                        section_list.append(sections[prev_end:current_end])

                        paper_embedding = torch.cat(embeddings,axis=0)
                        paper_importances = torch.cat(importances)
                        torch.save(paper_embedding, os.path.join(args.output_path, prev_id+".embedding"))
                        torch.save(paper_importances, os.path.join(args.output_path, prev_id+".importance"))

                        prev_end = i
                        
                    prev_id = paper_id
                    embeddings = []
                    importances = []
                    paper_id_list = []
                    section_list = []
            embeddings.append(pooler_output[current_end:])
            importances.append(importance[current_end:])
            paper_id_list.append(paper_ids[current_end:])
            section_list.append(sections[current_end:])

        paper_embedding = torch.cat(embeddings,axis=0)
        paper_importances = torch.cat(importances)
        torch.save(paper_embedding, os.path.join(args.output_path, prev_id+".embedding"))
        torch.save(paper_importances, os.path.join(args.output_path, prev_id+".importance"))