import os
import random
from glob import glob
from tqdm import tqdm
import argparse
from collections import OrderedDict

import numpy as np
import pandas as pd

import torch
from torch import cuda
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    T5Tokenizer,
    T5ForConditionalGeneration,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    BeamSearchScorer
)

from util import *

def load_t5_model(model_name, checkpoint=None):
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    if checkpoint is not None:
        try:
            model.load_state_dict(torch.load(checkpoint))
        except:
            old_state = torch.load(checkpoint)
            new_dict = OrderedDict()
            for k,v in old_state.items():
                new_dict[k[7:]] = v # Remove "module." prefix.
            model.load_state_dict(new_dict)

    #model = torch.nn.DataParallel(model)
    return model

class EmbeddingTitleDataset(Dataset):
    def __init__(self, t5_tokenizer, augment_tags=False, train = True, 
                 context_window = 2, MAX_SENT_LEN=9999, importance_rank = 0, embedding_size = 768,
                 cited_metadata_path = '/home/data/XiangciLi/20200705v1/acl/selected_cited_metadata.jsonl',
                 cached_embedding_path = '/home/data/XiangciLi/20200705v1/acl/embeddings'
                ):
        self.embedding_size = embedding_size
        self.max_sent_len = MAX_SENT_LEN
        self.context_window = context_window
        self.t5_tokenizer = t5_tokenizer
        self.importance_rank = importance_rank
        self.cached_embedding_path = cached_embedding_path

        cited_metadata = read_related_work_jsons(cited_metadata_path)
        self.samples = []
        embedding_list = glob(os.path.join(cached_embedding_path,"*.embedding"))
        for embedding_file in embedding_list:
            head, tail = os.path.split(embedding_file)
            paper_id = tail.split(".")[0]
            dictionary = cited_metadata[paper_id]
            self.samples.append(
                {
                    "sample_id": paper_id,
                    "source": "Title: ",
                    "target": dictionary["title"],
                    "citation_links": paper_id
                }
            )
        if train:
            self.samples = self.samples[:500]
        else:
            self.samples = self.samples[500:550]
                                    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ctext = self.samples[idx]["source"]

        text = self.samples[idx]["target"]

        source = self.t5_tokenizer.batch_encode_plus([ctext], max_length= 512, pad_to_max_length=True,return_tensors='pt')
        target = self.t5_tokenizer.batch_encode_plus([text], max_length= 512, pad_to_max_length=True,return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()
        
        citation_links = self.samples[idx]["citation_links"].split()
        paper_embeddings = []
        for link in citation_links:
            embedding_path = os.path.join(self.cached_embedding_path, link + ".embedding")
            if os.path.exists(embedding_path):
                embedding = torch.load(embedding_path)
                importance = torch.load(embedding_path.replace(".embedding",".importance"))
                selected_embedding = embedding[importance <= self.importance_rank]
                paper_embeddings.append(selected_embedding)
            else:
                paper_embeddings.append(torch.zeros(0, 512, self.embedding_size))
        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long),
            'paper_embeddings': paper_embeddings
        }

class EmbeddingCitationTextGenerationDataset(Dataset):
    def __init__(self, path_name: str, tokenizer, t5_tokenizer, augment_tags=False, train = True, 
                 context_window = 2, MAX_SENT_LEN=512, importance_rank = 2, embedding_size = 768,
                 related_work_path = '/home/data/XiangciLi/20200705v1/acl/selected_related_work.jsonl', 
                 cited_metadata_path = '/home/data/XiangciLi/20200705v1/acl/selected_cited_metadata.jsonl',
                 cached_embedding_path = '/home/data/XiangciLi/20200705v1/acl/embeddings'
                ):
        self.embedding_size = embedding_size
        self.max_sent_len = MAX_SENT_LEN
        self.context_window = context_window
        self.t5_tokenizer = t5_tokenizer
        self.importance_rank = importance_rank
        self.cached_embedding_path = cached_embedding_path
        self.discourse_label_types = {"Intro": 0,
        "Single_summ": 1,
        "Multi_summ": 2,
        "Narrative_cite":3,
        "Reflection":4,
        "Transition":5,
        "Other":6
        }
        
        discourse_tokens = []
        for k,v in self.discourse_label_types.items():
            discourse_tokens.append("["+k+"]")
        
        text_files = glob(os.path.join(path_name,"*.txt"))
        if len(text_files) == 0:
            text_files = []
            paths = glob(os.path.join(path_name,"*"))
            for path in paths:
                text_files.extend(glob(os.path.join(path,"*.txt")))
        
        self.related_work_jsons = read_related_work_jsons(related_work_path)
        self.cited_metadata = read_related_work_jsons(cited_metadata_path)
        
        self.dataset = {}
        for text_file in text_files:
            head, tail = os.path.split(text_file)
            paper_id = tail.split(".")[0]
            paragraphs, offsets, paragraph_begins = read_paragraphs_split(text_file, tokenizer, self.max_sent_len, 0)
            
            paragraph_ids = []
            pi = 0
            for b in paragraph_begins:
                if b:
                    part_id = 0
                    paragraph_ids.append(paper_id+ "_" + str(pi)+"_"+str(part_id))
                    pi+=1
                else:
                    part_id += 1
                    paragraph_ids.append(paper_id+ "_" + str(pi-1)+"_"+str(part_id))
                    
            annotation_file = text_file.replace(".txt",".ann")
            all_annotations = read_annotations(annotation_file, offsets)
            for paragraph_id, paragraph, paragraph_annotation in zip(paragraph_ids, paragraphs, all_annotations):
                for annotation in paragraph_annotation:
                    assert paragraph[annotation[0]:annotation[1]] == annotation[-1]
                sentences = [tokenizer.convert_tokens_to_string(tokenizer.tokenize(sent)) for sent in paragraph.split("[BOS] ")[1:]]
                tokens = tokenizer.tokenize(paragraph)
                N_tokens = len(tokens)
                discourse_labels = read_discourse_labels(paragraph_annotation, paragraph, self.discourse_label_types)
                #validate_span_annotation(paragraph_annotation)
                span_indices = read_span_indices(paragraph_annotation, paragraph)
                span_BIO_labels = get_span_BIO_labels(span_indices, paragraph, tokenizer)[1:-1]
                citation_mark_span_indices = read_citation_mark(paragraph_annotation, paragraph)
                citation_BIO_labels = get_citation_BIO_labels(citation_mark_span_indices, paragraph, tokenizer)[1:-1]

                #print(tokenizer.tokenize(paragraph))
                assert(N_tokens == len(span_BIO_labels) == len(citation_BIO_labels))
                #if not N_tokens == len(span_BIO_labels) == len(citation_BIO_labels):
                #    print(N_tokens, len(span_BIO_labels), len(citation_BIO_labels), paragraph_id, "Skipped")
                #    continue

                augmented_paragraph, augmented_sentences = make_augmented_paragraphs(tokens, tokenizer, discourse_tokens, discourse_labels, span_BIO_labels, citation_BIO_labels)
                paragraph_citation_links = sentence_citation_link(paragraph_id, augmented_sentences, self.related_work_jsons, tokenizer)
                span_sent_mapping, i_span = span_sentence_map(augmented_sentences)
                paragraph_citation_links = propagate_citation_cross_sentences(span_sent_mapping, paragraph_citation_links, i_span)
                self.dataset[paragraph_id] = {
                    "paragraph_id": paragraph_id,
                    "related_work":  paragraph, #augmented_paragraph,
                    "citation_links": paragraph_citation_links,
                    "augmented_sentences": augmented_sentences,
                    "discourse_labels": discourse_labels,
                    "sentences": sentences
                }
                
        self.samples = []
        for paragraph_id, paragraph in self.dataset.items():
            for si, (sent, links, discourse) in enumerate(zip(paragraph["sentences"], paragraph["citation_links"], paragraph["discourse_labels"])):
                if discourse == "Single_summ":
                    if augment_tags:
                        context_before = mount_discourse_sentence(paragraph["discourse_labels"][max(0,si-self.context_window):si], paragraph["augmented_sentences"][max(0,si-self.context_window):si])
                        context_after = mount_discourse_sentence(paragraph["discourse_labels"][si+1: si+self.context_window+1], paragraph["augmented_sentences"][si+1:: si+self.context_window+1])
                    else:
                        context_before = " ".join(paragraph["sentences"][max(0,si-self.context_window):si])
                        context_after = " ".join(paragraph["sentences"][si+1:: si+self.context_window+1])

                    context = context_before + " ["+ discourse +"]" + " [Answer] " + context_after
                    
                    if augment_tags:
                        dominant_context = ""
                        citation_links = []
                        for citation_mark, link in links["Dominant"].items():
                            if str(link) in self.cited_metadata:
                                cited_context = self.cited_metadata[str(link)]["abstract"]
                                if cited_context is None:
                                    cited_context = ""
                            else:
                                cited_context = ""
                            citation_links.append(str(link))
                            dominant_context += "[B_Dominant_context] "+citation_mark+" [SEP] " + cited_context + " [E_Dominant_context] "
                        reference_context = ""
                        for citation_mark, link in links["Reference"].items():
                            if str(link) in self.cited_metadata:
                                cited_context = self.cited_metadata[str(link)]["title"]
                                if cited_context is None:
                                    cited_context = ""
                            else:
                                cited_context = ""
                            citation_links.append(str(link))
                            reference_context += "[B_Reference_context] "+citation_mark+" [SEP] " + cited_context + " [E_Reference_context] "

                        source = "[B_Context] " + context + " [E_Context] " + dominant_context + " " + reference_context
                    else:
                        citation_context = ""
                        citation_links = []
                        merged_links = {**links["Dominant"], **links["Reference"]}
                        for citation_mark, link in merged_links.items():
                            citation_context += "[B_Citation] " + citation_mark + " [E_Citation] "
                            citation_links.append(str(link))
                        source = "[B_Context] " + context + " [E_Context] " + citation_context
                        
                    self.samples.append(
                        {
                            "sample_id": paragraph_id + "_" + str(si),
                            "source": "Generate: " + source,
                            "target": sent.strip(),
                            "citation_links": " ".join(citation_links)
                        }
                    )

                                    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ctext = self.samples[idx]["source"]

        text = self.samples[idx]["target"]

        source = self.t5_tokenizer.batch_encode_plus([ctext], max_length= 512, pad_to_max_length=True,return_tensors='pt')
        target = self.t5_tokenizer.batch_encode_plus([text], max_length= 512, pad_to_max_length=True,return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()
        
        citation_links = self.samples[idx]["citation_links"].split()
        paper_embeddings = []
        for link in citation_links:
            embedding_path = os.path.join(self.cached_embedding_path, link + ".embedding")
            if os.path.exists(embedding_path):
                embedding = torch.load(embedding_path)
                importance = torch.load(embedding_path.replace(".embedding",".importance"))
                selected_embedding = embedding[importance <= self.importance_rank]
                paper_embeddings.append(selected_embedding)
            else:
                paper_embeddings.append(torch.zeros(0,self.max_sent_len, self.embedding_size))
        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long),
            'paper_embeddings': paper_embeddings
        }
    
class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y
    
class TimeDistributedDense(nn.Module):
    def __init__(self, INPUT_SIZE, OUTPUT_SIZE):
        super(TimeDistributedDense, self).__init__()
        self.input_size = INPUT_SIZE
        self.output_size = OUTPUT_SIZE
        self.linear = nn.Linear(INPUT_SIZE, OUTPUT_SIZE, bias=True)
        self.timedistributedlayer = TimeDistributed(self.linear)
    def forward(self, x):
        # x: (BATCH_SIZE, ARRAY_LEN, INPUT_SIZE)
        
        return self.timedistributedlayer(x)

class WordAttention(nn.Module):
    """
    x: (BATCH_SIZE, N_sentence, N_token, INPUT_SIZE)
    out: (BATCH_SIZE, N_sentence, INPUT_SIZE)
    """
    def __init__(self, INPUT_SIZE, PROJ_SIZE, dropout = 0):
        super(WordAttention, self).__init__()
        self.activation = torch.tanh
        self.att_proj = TimeDistributedDense(INPUT_SIZE, PROJ_SIZE)
        self.dropout = nn.Dropout(dropout)
        self.att_scorer = TimeDistributedDense(PROJ_SIZE, 1)
        
    def forward(self, x):
        proj_input = self.att_proj(self.dropout(x.view(-1, x.size(-1))))
        proj_input = self.dropout(self.activation(proj_input))
        raw_att_scores = self.att_scorer(proj_input).squeeze(-1).view(x.size(0),x.size(1),x.size(2)) # (Batch_size, N_sentence, N_token)
        att_scores = F.softmax(raw_att_scores, dim=-1)
        att_scores = torch.where(torch.isnan(att_scores), torch.zeros_like(att_scores), att_scores) # Replace NaN with 0
        batch_att_scores = att_scores.view(-1, att_scores.size(-1)) # (Batch_size * N_sentence, N_token)
        out = torch.bmm(batch_att_scores.unsqueeze(1), x.view(-1, x.size(2), x.size(3))).squeeze(1) 
        # (Batch_size * N_sentence, INPUT_SIZE)
        out = out.view(x.size(0), x.size(1), x.size(-1))
        return out
    
class CrossHardAttention(nn.Module):
    def __init__(self, embedding_dim, query_dim, max_sent_len, merge=False):
        super(CrossHardAttention, self).__init__()
        self.merge = merge
        self.query_dim = query_dim
        self.max_sent_len = max_sent_len
        self.embedding_projection = TimeDistributedDense(embedding_dim, query_dim)
    def forward(self,paper_embedding, query_embedding):
        paper_batch = paper_embedding.shape[0]
        projected_paper = self.embedding_projection(paper_embedding)
        attention_matrix = torch.bmm(projected_paper, torch.tile(query_embedding.transpose(1,2),(paper_batch,1,1)))
        attention_score = torch.max(attention_matrix,dim=-1).values.view(-1)
        if self.merge:
            query_attention_score = torch.max(torch.max(attention_matrix,dim=-2).values.view(paper_batch,-1),dim=0).values
            attention_score = torch.cat([attention_score, query_attention_score])
            top_indices = torch.sort(torch.topk(attention_score,self.max_sent_len).indices).values
            flattened_embedding = projected_paper.view(-1, self.query_dim)
            flattened_embedding = torch.cat([flattened_embedding, query_embedding.view(-1,self.query_dim)], dim=0)
        else:
            top_indices = torch.sort(torch.topk(attention_score,self.max_sent_len).indices).values
            flattened_embedding = projected_paper.view(-1, self.query_dim)
        return flattened_embedding[top_indices].unsqueeze(0)
    
class MargeLayer(nn.Module):
    def __init__(self, embedding_dim, query_dim):
        super(MargeLayer, self).__init__()
        self.query_dim = query_dim
        self.embedding_projection = TimeDistributedDense(embedding_dim, query_dim//2)
        self.query_projection = TimeDistributedDense(query_dim, query_dim//2)
    def forward(self,paper_embedding, query_embedding):
        projected_paper = self.embedding_projection(paper_embedding.squeeze(0))
        projected_query = self.embedding_projection(query_embedding)
        return torch.cat([projected_paper, projected_query], dim=-1)
    
class MultiDocumentT5(nn.Module):
    def __init__(self, t5_path, paper_embedding_size, t5_checkpoint=None):
        super(MultiDocumentT5, self).__init__()
        self.num_beams = 5
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        self.t5_model = load_t5_model(t5_path, checkpoint=t5_checkpoint)
        #self.embedding_projection = nn.Linear(paper_embedding_size, self.t5_model.config.d_model, bias=True)
        #self.cross_hard_attention = CrossHardAttention(paper_embedding_size, self.t5_model.config.d_model, 512, True)
        self.merge_layer = MargeLayer(paper_embedding_size, self.t5_model.config.d_model)
        self.dummy_input = torch.ones((self.num_beams, 1), device=self.device, dtype=torch.long) * self.t5_model.config.decoder_start_token_id

        #self.word_attention = WordAttention(paper_embedding_size, self.t5_model.config.d_model)
        self.extra_modules = [
            self.merge_layer
            #self.cross_hard_attention
            #self.embedding_projection
            #self.word_attention
        ]
    
    def forward(self, input_ids=None, attention_mask=None, paper_embeddings = None, 
                query_segment_lens = None, embedding_lens = None, labels = None, inference=False):
        query_enc = self.t5_model.get_encoder()(input_ids = input_ids, attention_mask = attention_mask)
        #projected_embedding = self.embedding_projection(paper_embeddings)
        #projected_activation = self.embedding_projection(query_enc.last_hidden_state)
        #projected_embedding = paper_embeddings
        #if len(embedding_lens) > 0:
        #    paper_embeddings_list = torch.split(projected_embedding, embedding_lens, dim=1)
        #else:
        #    paper_embeddings_list = [projected_embedding]
       
        #segments = torch.split(query_enc.last_hidden_state, query_segment_lens, dim=1)
        
        #new_segments = [segments[0]]
        #for segment, projected_embedding in zip(segments[1:], paper_embeddings_list):
        #    projected_embedding = self.word_attention(projected_embedding)
        #    new_segments.append(projected_embedding)
        #    new_segments.append(segment[:,1:,:]) # Replace [Embedding] with the actual paper embedding.
        #combined = torch.cat(new_segments,dim=1)
        #if paper_embeddings.numel() > 0:
        #    concat_rep = self.cross_hard_attention(paper_embeddings.squeeze(0), query_enc.last_hidden_state)
        #else:
        #    concat_rep = query_enc.last_hidden_state
        
        if paper_embeddings.numel() > 0:
            concat_rep = self.merge_layer(paper_embeddings, query_enc.last_hidden_state)
        else:
            concat_rep = query_enc.last_hidden_state
        
        if inference:
            query_enc.last_hidden_state = concat_rep.repeat_interleave(self.num_beams, dim=0)
            model_kwargs = {
                "encoder_outputs": query_enc
            }
            
            beam_scorer = BeamSearchScorer(
                batch_size=1,
                max_length = 512,
                num_beams=self.num_beams,
                device=self.device
            )

            
            # instantiate logits processors
            logits_processor = LogitsProcessorList([
                MinLengthLogitsProcessor(15, eos_token_id=self.t5_model.config.eos_token_id),
            ])
            #outputs = self.t5_model.greedy_search(self.dummy_input, logits_processor=logits_processor, **model_kwargs)
            outputs = self.t5_model.beam_search(self.dummy_input, beam_scorer, logits_processor=logits_processor, **model_kwargs)
        else:
            query_enc.last_hidden_state = concat_rep
            outputs = self.t5_model(encoder_outputs=query_enc, labels = labels)
        return outputs
    
def train(epoch, tokenizer, model, device, loader, optimizer, args, additional_special_token_mapping):
    torch.cuda.empty_cache()
    model.train()
    tq = tqdm(loader)
    losses = []
    for _,data in enumerate(tq):
        input_ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)
        y = data['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100

        position = torch.where(input_ids == additional_special_token_mapping["[Embedding]"])
        foo1 = torch.cat([position[1].cpu(), torch.tensor([args.max_sent_len])])
        foo2 = torch.cat([torch.tensor([0]), position[1].cpu()])
        segment_lens = (foo1 - foo2).tolist()

        embedding_lens = []
        for embedding in data["paper_embeddings"]:
            if embedding is not None:
                embedding_lens.append(embedding.shape[1])

        if len(data["paper_embeddings"]) > 0:
            paper_embeddings = torch.cat(data["paper_embeddings"],dim=1).to(device)
        else:
            paper_embeddings = torch.zeros((1, 0, args.max_sent_len, args.paper_embedding_dim)).to(device)

        outputs = model(input_ids=input_ids, attention_mask=mask, query_segment_lens = segment_lens, 
            paper_embeddings = paper_embeddings, embedding_lens = embedding_lens, labels = lm_labels)

        loss = outputs[0]
        losses.append(loss.sum().item())
        tq.set_description(f'Epoch {epoch}, iter {_}, loss: {np.mean(losses)}')

        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()
        
def evaluate(epoch, tokenizer, model, device, loader, args, additional_special_token_mapping):
    torch.cuda.empty_cache()
    model.train()
    tq = tqdm(loader)
    losses = []
    for _,data in enumerate(tq):
        input_ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)
        y = data['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100

        position = torch.where(input_ids == additional_special_token_mapping["[Embedding]"])
        foo1 = torch.cat([position[1].cpu(), torch.tensor([args.max_sent_len])])
        foo2 = torch.cat([torch.tensor([0]), position[1].cpu()])
        segment_lens = (foo1 - foo2).tolist()

        embedding_lens = []
        for embedding in data["paper_embeddings"]:
            if embedding is not None:
                embedding_lens.append(embedding.shape[1])

        if len(data["paper_embeddings"]) > 0:
            paper_embeddings = torch.cat(data["paper_embeddings"],dim=1).to(device)
        else:
            paper_embeddings = torch.zeros((1,0,args.paper_embedding_dim)).to(device)

        outputs = model(input_ids=input_ids, attention_mask=mask, query_segment_lens = segment_lens, 
            paper_embeddings = paper_embeddings, embedding_lens = embedding_lens, labels = lm_labels)

        loss = outputs[0]
        losses.append(loss.sum().item())
        tq.set_description(f'Epoch {epoch}, iter {_}, loss: {np.mean(losses)}')
    
def validate(tokenizer, model, device, loader, args, additional_special_token_mapping):
    model.eval()
    predictions = []
    actuals = []
    sources = []
    with torch.no_grad():
        tq = tqdm(loader)
        for _, data in enumerate(tq, 0): 
            if _ >= 100:
                break
            y = data['target_ids'].to(device, dtype = torch.long)
            input_ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)
            position = torch.where(input_ids == additional_special_token_mapping["[Embedding]"])
            foo1 = torch.cat([position[1].cpu(), torch.tensor([args.max_sent_len])])
            foo2 = torch.cat([torch.tensor([0]), position[1].cpu()])
            segment_lens = (foo1 - foo2).tolist()
            embedding_lens = []
            for embedding in data["paper_embeddings"]:
                if embedding is not None:
                    embedding_lens.append(embedding.shape[1])
                    
            if len(data["paper_embeddings"]) > 0:
                paper_embeddings = torch.cat(data["paper_embeddings"],dim=1).to(device)
            else:
                paper_embeddings = torch.zeros((1,0,args.paper_embedding_dim)).to(device)
                    
            generated_ids = model(input_ids=input_ids, attention_mask=mask, query_segment_lens = segment_lens, 
            paper_embeddings = paper_embeddings, embedding_lens = embedding_lens, inference=True)
            
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            source = [tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True)for s in input_ids]
            predictions.extend(preds)
            actuals.extend(target)
            sources.extend(source)
    return predictions, actuals, sources
    
def main(args):
    device = 'cuda' if cuda.is_available() else 'cpu'
    
    discourse_label_types = {"Intro": 0,
    "Single_summ": 1,
    "Multi_summ": 2,
    "Narrative_cite":3,
    "Reflection":4,
    "Transition":5,
    "Other":6
    }
    
    bert_path = "allenai/scibert_scivocab_uncased"
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_path)

    special_tokens = ['[BOS]', '[B_span]', '[E_span]', '[B_Dominant]', '[I_Dominant]', '[E_Dominant]', '[B_Reference]', '[I_Reference]','[E_Reference]', '[O]']
    discourse_tokens = []
    for k,v in discourse_label_types.items():
        discourse_tokens.append("["+k+"]")
    special_tokens.extend(discourse_tokens)

    additional_special_tokens = {'additional_special_tokens': special_tokens}
    bert_tokenizer.add_special_tokens(additional_special_tokens)
    
    tokenizer = T5Tokenizer.from_pretrained(args.repfile)
    special_tokens.extend(['[B_Context]', '[E_Context]', '[SEP]', '[Embedding]','[B_Citation]', '[E_Citation]','[B_Dominant_context]', '[E_Dominant_context]', '[B_Reference_context]', '[E_Reference_context]'])
    tokenizer.add_special_tokens(additional_special_tokens)
    additional_special_token_mapping = {k:v for k,v in zip(tokenizer.additional_special_tokens, tokenizer.additional_special_tokens_ids)}
    
    #training_set = EmbeddingCitationTextGenerationDataset(args.train_file, bert_tokenizer, tokenizer, embedding_size = args.paper_embedding_dim)
    #dev_set = EmbeddingCitationTextGenerationDataset(args.test_file, bert_tokenizer, tokenizer, embedding_size = args.paper_embedding_dim)
    
    training_set = EmbeddingTitleDataset(tokenizer, train=True, cached_embedding_path = args.cached_embedding_path, embedding_size = args.paper_embedding_dim)
    dev_set = EmbeddingTitleDataset(tokenizer, train=False, cached_embedding_path = args.cached_embedding_path, embedding_size = args.paper_embedding_dim)
    
    model = MultiDocumentT5(args.repfile, args.paper_embedding_dim, t5_checkpoint = args.pre_trained_t5).to(device)
    if args.pre_trained_model is not None:
        model.load_state_dict(torch.load(args.pre_trained_model))
    
    settings = [{'params': model.t5_model.parameters(), 'lr': args.lr}]
    for module in model.extra_modules:
        settings.append({'params': module.parameters(), 'lr': args.lr})
    optimizer = torch.optim.Adam(settings)
    
    train_params = {
        'batch_size': args.batch_size,
        'shuffle': False, 
        'num_workers': 0
    }
    
    val_params = {
        'batch_size': args.batch_size,
        'shuffle': False,
        'num_workers': 0
    }
    
    train_loader = DataLoader(training_set, **train_params)
    dev_loader = DataLoader(dev_set, **val_params)
    
    for epoch in range(args.epoch):
        train(epoch, tokenizer, model, device, train_loader, optimizer, args, additional_special_token_mapping)
        torch.save(model.state_dict(), args.checkpoint+"_"+str(epoch)+".model")
        evaluate(epoch, tokenizer, model, device, dev_loader, args, additional_special_token_mapping)
        
    predictions, actuals, sources = validate(tokenizer, model, device, dev_loader, args, additional_special_token_mapping)
    final_df = pd.DataFrame({'Generated Text':predictions,'Actual Text':actuals, 'Source Text': sources})
    final_df.to_csv('./predictions.csv')
    print('Output Files generated for review')
        
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train, cross-validate and run sentence sequence tagger")
    argparser.add_argument('--repfile', type=str, default = "t5-base", help="Word embedding file")
    argparser.add_argument('--train_file', type=str, default="")
    argparser.add_argument('--pre_trained_t5', type=str)
    argparser.add_argument('--pre_trained_model', type=str)
    argparser.add_argument('--test_file', type=str, default="")
    argparser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    argparser.add_argument('--paper_embedding_dim', type=int, default=768, help="bert_dimension")
    argparser.add_argument('--epoch', type=int, default=15, help="Training epoch")
    argparser.add_argument('--max_sent_len', type=int, default=512)
    argparser.add_argument('--checkpoint', type=str, default = "paper_embedding_t5")
    argparser.add_argument('--log_file', type=str, default = "paper_embedding_t5_performances.jsonl")
    argparser.add_argument('--batch_size', type=int, default=1) # roberta-large: 2; bert: 8
    argparser.add_argument('--cached_embedding_path', type=str, default = "paper_embedding_t5")
    
    args = argparser.parse_args()
    main(args)