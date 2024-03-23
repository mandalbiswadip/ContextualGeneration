import os
import sys
from glob import glob
from util import read_paragraphs, validate_span_annotation, read_discourse_labels

discourse_label_types = {"Intro": 0,
"Single_summ": 1,
"Multi_summ": 2,
"Narrative_cite":3,
"Reflection":4,
"Transition":5,
"Other":6
}

def read_annotations(annotation_file):
    annotations = []
    with open(annotation_file) as f:
        for line in f:
            ID, content, text = line.strip().split("\t")
            content = content.split()
            label = content[0]
            start = int(content[1])
            end = int(content[2])            
            annotations.append((start, end, label, text))
        annotations = sorted(annotations, key=lambda x: x[0])
    return annotations


text_files = glob(os.path.join(sys.argv[1],"*.txt"))
all_annotations = {}
for text_file in text_files:
    head, tail = os.path.split(text_file)
    paper_id = tail.split(".")[0]
    paragraphs, offsets = read_paragraphs(text_file)
    #print(paper_id, len("".join(paragraphs).strip().split()))
    print("Validating paper",paper_id)
    annotation_file = text_file.replace(".txt",".ann")
    annotations = read_annotations(annotation_file)
    all_annotations[paper_id] = annotations
    validate_span_annotation(annotations)
    read_discourse_labels(annotations, "".join(paragraphs), discourse_label_types)
print("Congrats, all annotations are valid!")