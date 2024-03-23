"""
Add citation labels to .ann files for visualizing in BART
NOTE: THIS WILL APPEND ON THE EXISTING .ANN FILES
"""
import argparse
import json
import logging
import os
from glob import glob

from config import classification_tasks


def get_ann_path(ann_folder, scires_file):
    """
    Get .ann file path from .scires file.
    target folder path for ann files is required
    """
    return os.path.join(
        ann_folder,
        os.path.basename(os.path.splitext(scires_file)[0]) + ".ann"
    )


def load_citation_file(file):
    with open(file, "r") as reader:
        con = json.load(reader)
    return con


def get_line_counter(ann_file):
    with open(ann_file, "r") as reader:
        return int(reader.read().splitlines()[-1].split("\t")[0][1:])


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Predict citation functions!")
    argparser.add_argument('--citation_type', type=str)
    argparser.add_argument('--citation_file', type=str)
    argparser.add_argument('--corwa_file', type=str)
    logging.getLogger("transformers.tokenization_utils_base").setLevel(
        logging.ERROR)

    args = argparser.parse_args()

    params = vars(args)

    for k, v in params.items():
        print(k, v)

    if args.citation_type not in classification_tasks:
        raise ValueError(
            "citation_type should be any of {}".format(classification_tasks))

    citation_file = args.citation_file

    for file in glob(
            os.path.join(citation_file, "*.{}".format(args.citation_type))):

        citation_data = load_citation_file(file)
        ann_file = get_ann_path(args.corwa_file, file)

        with open(ann_file, "a") as reader:
            count_lines = get_line_counter(ann_file)
            count_lines += 200
            lines = []
            for row in citation_data:
                line = "T{}\t{} {} {}\t{}".format(
                    count_lines, row["{}_label".format(args.citation_type)],
                    row["start"],
                    row["end"], row["citation_text"]
                )

                lines.append(line + "\n")
                count_lines += 1

            if len(lines) > 0:
                # lines[-1] = lines[-1][:-1]
                reader.writelines(lines)
