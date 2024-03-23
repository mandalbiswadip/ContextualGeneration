import argparse

from util import read_passages_json
import jsonlines
import json

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train, cross-validate and run sentence sequence tagger")
    argparser.add_argument('--output_jsonl', type=str, default = "related_work_discourse_train.jsonl")
    argparser.add_argument('--merge_jsonl', nargs='+', help='<Required> Set flag', required=True)
    args = argparser.parse_args()

    with jsonlines.open(args.output_jsonl,"w") as wf:
        for file_name in args.merge_jsonl:
            with open(file_name) as f:
                for line in f:
                    json_dict = json.loads(line)
                    wf.write(json_dict)