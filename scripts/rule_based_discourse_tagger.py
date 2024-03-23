import argparse

import json
from tqdm import tqdm
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize

import re
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
    
def tsv_record(array):
    string = ""
    for element in array:
        string += str(element) + "\t"
    return string[:-1]+"\n"

def has_keyword(keywords, sentence):
    for kw in keywords:
        if kw in sentence:
            return True
    return False

def contain_citation(sentence):
    matches = re.findall("\[[0-9,]*\]|\([a-zA-Z.& ]+,[^A-Za-z0-9_]?[0-9]*[a-zA-Z0-9.&,; ]*\)|[A-Z][a-zA-Z. ]* \([0-9]*\)", sentence)
    return len(matches)

def contain_year(sentence):
    matches = re.findall("19[0-9]{2}|20[0-9]{2}", sentence)
    return len(matches)

def tag_discourse(sentence, citation_count, pi, prev):
    lower_sentence = sentence.lower()
    if pi==0 and has_keyword(["in this", "we will", "to be"], lower_sentence):
        return "i"
    if has_keyword(["we ", "will ", " our ", "ours", "this paper"], lower_sentence) or has_keyword(["Our "], sentence):
        return "r"
    if citation_count > 1:
        return "n"
    if citation_count == 1:
        return "c"
    if prev=="c" and has_keyword(["they", "their", "author", " it ", " it's "], lower_sentence):
        return "c"
    return "t"

def patch_sent_tokenize(sentences):
    out = []
    i = 0
    while i < len(sentences):
        if i>0 and sentences[i-1][-4:] == " et." and sentences[i][:2] == "al":
            out[-1] += " " + sentences[i]
        elif i>0 and (sentences[i-1][-4:] == " al." or sentences[i-1]=="al."):
            out[-1] += " " + sentences[i]
        elif i>0 and sentences[i-1][-4:] == "e.g.":
            out[-1] += " " + sentences[i]
        elif i>0 and sentences[i-1][-4:] == "i.e.":
            out[-1] += " " + sentences[i]
        else:
            out.append(sentences[i])
        i += 1
    return out


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train, cross-validate and run sentence sequence tagger")
    argparser.add_argument('--related_work_file', type=str, default = "/home/xxl190027/20200705v1/acl/related_work.jsonl")
    argparser.add_argument('--output_file', type=str, default = "related_work_discourse_rule.tsv")
    args = argparser.parse_args()

    count = 0
    with open(args.related_work_file,"r") as f_pdf:
        with open(args.output_file,"w") as wf:
            for line in tqdm(f_pdf):
                related_work_dict = json.loads(line)
                for pi, para in enumerate(related_work_dict["related_work"]):
                    cite_span_texts = set([citation["text"] for citation in para["cite_spans"]])
                    sentences = []
                    citation_counts = []
                    tags = []
                    tag = ""
                    for si, sentence in enumerate(patch_sent_tokenize(sent_tokenize(para["text"]))):
                        citation_count = 0
                        for citation in cite_span_texts:
                            if citation in sentence:
                                citation_count+=1
                        if citation_count == 0: # Try to extract citation for the second time, in case S2ORC did not find them out.
                            citation_count = contain_citation(sentence)
                        if citation_count == 0:
                            citation_count = contain_year(sentence)
                        sentences.append(sentence)
                        tag = tag_discourse(sentence, citation_count, pi, tag)
                        tags.append(tag)
                        citation_counts.append(citation_count)
                    if "i" in tags:
                        if sum(citation_counts) == 0:
                            tags = ["i" for tag in tags]
                    for si, (sentence, count, tag) in enumerate(zip(sentences, citation_counts, tags)):
                        wf.write(tsv_record([related_work_dict["paper_id"],pi+1,si+1,sentence,count>0, tag]))
                    #for si, (sentence, count, tag) in enumerate(zip(sentences, citation_counts, tags)):
                    #    wf.write(tsv_record([sentence,tag]))
                    #wf.write("\n")