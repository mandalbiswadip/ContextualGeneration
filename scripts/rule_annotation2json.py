import argparse
import jsonlines
import json

head = 1
def read_skeleton(tsv_file, head=1): 
    pp_ids = []
    pp_ids_set = set([])
    prev_len = len(pp_ids_set)
    count = 0
    with open(tsv_file) as f:
        for line in f:
            count += 1
            if count > head:
                tokens = line.strip().split("\t")
                if len(tokens) > 6:
                    tokens = tokens[:6]
                paper_id, p_id, s_id, sentence, citation, label = tokens
                pp_id = paper_id+"_"+p_id
                pp_ids_set.add(pp_id)
                if len(pp_ids_set) > prev_len:
                    prev_len=len(pp_ids_set)
                    pp_ids.append(pp_id)
    everything = {pp_id:{"sentence":[], "label":[]} for pp_id in pp_ids}
    return everything

def read_sentences(tsv_file, everything, head=1):
    count = 0
    with open(tsv_file) as f:
        for line in f:
            count += 1
            if count > head:
                tokens = line.strip().split("\t")
                if len(tokens) > 6:
                    tokens = tokens[:6]
                paper_id, p_id, s_id, sentence, citation, label = tokens

                if sentence[0]=='"' and sentence[-1]=='"':
                    sentence = sentence[1:-1]
                pp_id = paper_id+"_"+p_id
                everything[pp_id]["sentence"].append(sentence)
                everything[pp_id]["label"].append(label)
    return everything

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train, cross-validate and run sentence sequence tagger")
    argparser.add_argument('--tsv_file', type=str, default = "related_work_discourse_rule.tsv")
    argparser.add_argument('--json_file', type=str, default = "related_work_discourse_train.jsonl")
    argparser.add_argument('--head', type=int, default = 1)
    args = argparser.parse_args()
    
    everything = read_skeleton(args.tsv_file, head= args.head)
    everything = read_sentences(args.tsv_file, everything, head= args.head)
    
    with jsonlines.open(args.json_file, 'w') as output:
        for k,v in everything.items():
            v["id"] = k
            output.write(v)