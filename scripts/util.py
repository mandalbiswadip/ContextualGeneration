import codecs
import copy
import json
import os
import random
import re
from collections import defaultdict

from nltk import sent_tokenize, word_tokenize
from sklearn.metrics import f1_score
import torch

def flatten(arrayOfArray):
    array = []
    for arr in arrayOfArray:
        try:
            array.extend(arr)
        except:
            array.append(arr)
    return array


def read_passages(filename, is_labeled):
    str_seqs = []
    str_seq = []
    label_seqs = []
    label_seq = []
    for line in codecs.open(filename, "r", "utf-8"):
        lnstrp = line.strip()
        if lnstrp == "":
            if len(str_seq) != 0:
                str_seqs.append(str_seq)
                str_seq = []
                label_seqs.append(label_seq)
                label_seq = []
        else:
            if is_labeled:
                clause, label = lnstrp.split("\t")
                label_seq.append(label.strip())
            else:
                clause = lnstrp
            str_seq.append(clause)
    if len(str_seq) != 0:
        str_seqs.append(str_seq)
        str_seq = []
        label_seqs.append(label_seq)
        label_seq = []
    return str_seqs, label_seqs


def read_passages_json(filename, is_labeled):
    str_seqs = []
    label_seqs = []
    paper_ids = []
    with open(filename) as f:
        for line in f:
            json_dict = json.loads(line)
            str_seqs.append(json_dict["sentence"])
            if is_labeled:
                label_seqs.append(json_dict["label"])
            paper_ids.append(json_dict["id"])
    return paper_ids, str_seqs, label_seqs


def from_BIO_ind(BIO_pred, BIO_target, indices):
    table = {}  # Make a mapping between the indices of BIO_labels and temporary original label indices
    original_labels = []
    for BIO_label, BIO_index in indices.items():
        if BIO_label[:2] == "I_" or BIO_label[:2] == "B_":
            label = BIO_label[2:]
        else:
            label = BIO_label
        if label in original_labels:
            table[BIO_index] = original_labels.index(label)
        else:
            table[BIO_index] = len(original_labels)
            original_labels.append(label)

    original_pred = [table[label] for label in BIO_pred]
    original_target = [table[label] for label in BIO_target]
    return original_pred, original_target


def to_BIO(label_seqs):
    new_label_seqs = []
    for label_para in label_seqs:
        new_label_para = []
        prev = ""
        for label in label_para:
            if label != "none":  # "none" is O, remain unchanged.
                if label == prev:
                    new_label = "I_" + label
                else:
                    new_label = "B_" + label
            else:
                new_label = label  # "none"
            prev = label
            new_label_para.append(new_label)
        new_label_seqs.append(new_label_para)
    return new_label_seqs


def from_BIO(label_seqs):
    new_label_seqs = []
    for label_para in label_seqs:
        new_label_para = []
        for label in label_para:
            if label[:2] == "I_" or label[:2] == "B_":
                new_label = label[2:]
            else:
                new_label = label
            new_label_para.append(new_label)
        new_label_seqs.append(new_label_para)
    return new_label_seqs


def clean_url(word):
    """
        Clean specific data format from social media
    """
    # clean urls
    word = re.sub(r'https? : \/\/.*[\r\n]*', '<URL>', word)
    word = re.sub(r'exlink', '<URL>', word)
    return word


def clean_num(word):
    # check if the word contain number and no letters
    if any(char.isdigit() for char in word):
        try:
            num = float(word.replace(',', ''))
            return '@'
        except:
            if not any(char.isalpha() for char in word):
                return '@'
    return word


def clean_words(str_seqs):
    processed_seqs = []
    for str_seq in str_seqs:
        processed_clauses = []
        for clause in str_seq:
            filtered = []
            tokens = clause.split()
            for word in tokens:
                word = clean_url(word)
                # word = clean_num(word)
                filtered.append(word)
            filtered_clause = " ".join(filtered)
            processed_clauses.append(filtered_clause)
        processed_seqs.append(processed_clauses)
    return processed_seqs


def test_f1(test_file, pred_label_seqs):
    def linearize(labels):
        linearized = []
        for paper in labels:
            for label in paper:
                linearized.append(label)
        return linearized

    _, label_seqs = read_passages_original(test_file, True)
    true_label = linearize(label_seqs)
    pred_label = linearize(pred_label_seqs)

    f1 = f1_score(true_label, pred_label, average="weighted")
    print("F1 score:", f1)
    return f1


def patch_sent_tokenize(sentences):
    out = []
    i = 0
    while i < len(sentences):
        if i > 0 and sentences[i - 1][-4:] == " et." and sentences[i][
                                                         :2] == "al":
            out[-1] += " " + sentences[i]
        elif i > 0 and (
                sentences[i - 1][-4:] == " al." or sentences[i - 1] == "al."):
            out[-1] += " " + sentences[i]
        elif i > 0 and sentences[i - 1][-4:] == "e.g.":
            out[-1] += " " + sentences[i]
        elif i > 0 and sentences[i - 1][-4:] == "i.e.":
            out[-1] += " " + sentences[i]
        else:
            out.append(sentences[i])
        i += 1
    return out


def read_paragraphs(text_file):
    with open(text_file) as f:
        raw_strings = f.readlines()
        all_text = "".join(raw_strings)

    offset = 0
    pointer = 0
    all_texts = []
    offsets = []
    sent_lens = [len(sent) for sent in
                 all_text.strip().split("\n")]  # The unit is char
    for sent_len in sent_lens:
        pointer += sent_len + 1
        if sent_len == 0:
            all_texts.append(all_text[offset:pointer])
            offsets.append(offset)
            offset = pointer
    all_texts.append(all_text[offset:])
    offsets.append(offset)
    return all_texts, offsets


def read_paragraphs_split_oldest(text_file, tokenizer, max_sent_len):
    with open(text_file) as f:
        raw_strings = f.readlines()
        all_text = "".join(raw_strings)

    offset = 0
    pointer = 0
    cummulative_token_len = 0
    all_texts = []
    offsets = []
    paragraph_begins = [True]
    sent_lens = [(len(sent), len(tokenizer.tokenize(sent))) for sent in
                 all_text.strip().split("\n")]  # Units in char and token
    for sent_len_char, sent_len_token in sent_lens:
        pointer += sent_len_char + 1
        cummulative_token_len += sent_len_token
        # An empty line is used to segment paragraphs.
        if sent_len_char == 0:
            all_texts.append(all_text[offset:pointer])
            offsets.append(offset)
            paragraph_begins.append(True)
            offset = pointer
            cummulative_token_len = 0

        elif cummulative_token_len >= max_sent_len - 1:  # More conservative
            previous_pointer = pointer - sent_len_char - 1
            all_texts.append(all_text[offset:previous_pointer])
            offsets.append(offset)
            paragraph_begins.append(False)
            offset = previous_pointer
            cummulative_token_len = sent_len_token
            # Assume one single sentence is not too long to exceed max_sent_len

    all_texts.append(all_text[offset:])
    offsets.append(offset)
    return all_texts, offsets, paragraph_begins


def read_paragraphs_split_roberta(text_file, tokenizer, max_sent_len):
    bos_token_id = tokenizer("[BOS]")['input_ids'][1]
    line_breaker_id = tokenizer("\n\n")['input_ids'][1]
    with open(text_file) as f:
        raw_strings = f.readlines()
        all_text = "".join(raw_strings)

    all_texts = []
    offsets = [0]
    prev_token_id = -1
    paragraph_begins = [True]
    prev_paragraph_index = -1
    prev_sent_index = -1

    tokenized = tokenizer(all_text, return_offsets_mapping=True)
    for i, (token_id, offset_mapping) in enumerate(
            zip(tokenized["input_ids"], tokenized["offset_mapping"])):
        if token_id == line_breaker_id:
            all_texts.append(all_text[offsets[-1]:offset_mapping[1]])
            paragraph_begins.append(True)
            offsets.append(offset_mapping[1])
            prev_paragraph_index = i
        elif i - prev_paragraph_index >= max_sent_len - 1:
            all_texts.append(all_text[offsets[-1]:tokenized["offset_mapping"][
                prev_sent_index][0]])
            offsets.append(tokenized["offset_mapping"][prev_sent_index][0])
            paragraph_begins.append(False)
            prev_paragraph_index = prev_sent_index
        if token_id == bos_token_id:
            prev_sent_index = i
        prev_token_id = token_id
    if len(all_text[offsets[-1]:]) > 0:
        all_texts.append(all_text[offsets[-1]:])
    else:
        offsets = offsets[:-1]
        paragraph_begins = paragraph_begins[:-1]
    return all_texts, offsets, paragraph_begins


def read_paragraphs_split(text_file, tokenizer, max_sent_len):
    bos_token_id = tokenizer("[BOS]")['input_ids'][1]
    with open(text_file) as f:
        raw_strings = f.readlines()
        all_text = "".join(raw_strings).strip()

    all_texts = []
    offsets = []
    paragraph_begins = []
    paragraphs = [para + "\n\n" for para in all_text.split("\n\n")]
    if paragraphs[-1] == "\n\n":
        _ = paragraphs.pop()
    prev_paragraph_char_index = 0
    for paragraph in paragraphs:
        prev_segment_index = -1
        current_offset = 0
        prev_sent_index = -1
        prev_offset = 0
        tokenized = tokenizer(paragraph, return_offsets_mapping=True)
        paragraph_begins.append(True)
        offsets.append(prev_paragraph_char_index)
        for i, (token_id, offset_mapping) in enumerate(
                zip(tokenized["input_ids"], tokenized["offset_mapping"])):
            if i - prev_segment_index >= max_sent_len - 1:
                current_offset = tokenized["offset_mapping"][prev_sent_index][0]
                all_texts.append(paragraph[prev_offset:current_offset])
                offsets.append(current_offset + prev_paragraph_char_index)
                paragraph_begins.append(False)
                prev_segment_index = prev_sent_index
                prev_offset = current_offset
            if token_id == bos_token_id:
                prev_sent_index = i
        prev_paragraph_char_index += len(paragraph)
        all_texts.append(paragraph[current_offset:])
    return all_texts, offsets, paragraph_begins


def read_annotations(annotation_file, offsets):
    annotations = []
    with open(annotation_file) as f:
        for line in f:
            ID, content, text = line.strip().split("\t")
            content = content.split()
            label = content[0]
            if label == "Pronoun":
                continue
            start = int(content[1])
            end = int(content[2])
            annotations.append((start, end, label, text))
    annotations = sorted(annotations, key=lambda x: x[0])
    all_annotations = [[]]
    paragraph_index = 0
    offsets.append(1e10)  # A big number
    try:
        for start, end, label, text in annotations:
            if start >= offsets[paragraph_index + 1]:
                paragraph_index += 1
                all_annotations.append([])
            start -= offsets[paragraph_index]
            end -= offsets[paragraph_index]
            all_annotations[paragraph_index].append((start, end, label, text))
    except:
        print(annotation_file, offsets)
        assert()
    return all_annotations


def read_discourse_labels(annotations, all_text, discourse_label_types):
    N_sent = len(all_text.split("[BOS]")) - 1
    discourse_labels = []
    for ann in annotations:
        if "BOS" in ann[-1] and ann[-2] in discourse_label_types:
            discourse_labels.append(ann[-2])
    assert len(
        discourse_labels) == N_sent, "Please check if every [BOS] token is annotated."
    return discourse_labels


# def validate_span_annotation(annotations):
#    in_span = False
#    for ann in annotations:
#        if ann[-2] == 'B_span':
#            assert not in_span, "Missing E_span"
#            in_span = True
#        elif ann[-2] == 'E_span':
#            assert in_span, "Missing S_span"
#            in_span = False
#    assert not in_span, "Mssing the last E_span"

def validate_span_annotation(annotations):
    stack = []
    for ann in annotations:
        if ann[-2] == 'B_span':
            stack.append(True)
        elif ann[-2] == 'E_span':
            stack.pop()
    assert len(
        stack) == 0, "Please double check whether B_span and E_span form pairs"


def back_trace_chars(index, all_text):
    while index > 0 and (
            all_text[index - 1].isalnum() or all_text[index - 1] in {"<", "(",
                                                                     "["}):
        index -= 1
    return index


def forward_trace_chars(index, all_text):
    while index < len(all_text) - 1 and (
            all_text[index].isalnum() or all_text[index] in {".", ")", "]",
                                                             ">"}):
        index += 1
    return index


def read_span_indices(annotations, all_text):
    span_indices = []
    stack = []
    for ann in annotations:
        if ann[-2] == 'B_span':
            start = back_trace_chars(ann[0], all_text)
            stack.append(start)
        elif ann[-2] == 'E_span':
            end = forward_trace_chars(ann[1], all_text)
            start = stack.pop()
            span_indices.append((start, end, "span"))

    return sorted(span_indices, key=lambda x: x[0])


def get_span_BIO_labels(span_indices, all_text, tokenizer):
    """
    Strategy: Tokenize segment by segment to ensure the token and label matches. OOO|BIIII|OOO|BIII|OOO
    """
    prev_end = 0
    span_labels = []
    for start, end in span_indices:
        if start < prev_end:
            continue
        out_of_span_tokens_len = len(
            tokenizer.tokenize(all_text[prev_end: start]))
        span_labels.extend(["O"] * out_of_span_tokens_len)
        in_span_tokens_len = len(tokenizer.tokenize(all_text[start: end]))
        span_labels.extend(["B_span"] + ["I_span"] * (in_span_tokens_len - 1))
        prev_end = end
    out_of_span_tokens_len = len(tokenizer.tokenize(all_text[prev_end:]))
    span_labels.extend(["O"] * out_of_span_tokens_len)
    return ["O"] + span_labels + ["O"]  # CLS and SEP tokens


def get_span_BIOES_labels(span_indices, all_text, tokenizer):
    """
    Strategy: Tokenize segment by segment to ensure the token and label matches. OOO|BIBIIEIE|OOO|BIIE|OOO
    Some annoying conflict may occur when two S_span or E_span overlaps.
    """
    prev_token_start = 0
    prev_start = 0
    prev_end = 0
    span_labels = []
    for start, end in span_indices:
        if end <= prev_end:
            out_of_span_tokens_len = len(
                tokenizer.tokenize(all_text[prev_start: start]))
            in_span_tokens_len = len(tokenizer.tokenize(all_text[start: end]))
            if in_span_tokens_len >= 2:
                span_labels[
                    prev_token_start + out_of_span_tokens_len] = "B_span"
                span_labels[
                    prev_token_start + out_of_span_tokens_len + in_span_tokens_len - 1] = "E_span"
            else:
                span_labels.append("S_span")
        else:
            out_of_span_tokens_len = len(
                tokenizer.tokenize(all_text[prev_end: start]))
            span_labels.extend(["O"] * out_of_span_tokens_len)
            in_span_tokens_len = len(tokenizer.tokenize(all_text[start: end]))
            prev_token_start = len(span_labels)
            if in_span_tokens_len >= 2:
                span_labels.extend(
                    ["B_span"] + ["I_span"] * (in_span_tokens_len - 2) + [
                        "E_span"])
            else:
                span_labels.append("S_span")
            prev_start = start
            prev_end = end
    out_of_span_tokens_len = len(tokenizer.tokenize(all_text[prev_end:]))
    span_labels.extend(["O"] * out_of_span_tokens_len)
    return ["O"] + span_labels + ["O"]  # CLS and SEP tokens


def read_citation_mark(annotations, all_text,
                       citation_annotation_types={"Dominant", "Reference"}):
    citation_mark_span_indices = []
    prev_end = 0
    for ann in annotations:
        if ann[-2] in citation_annotation_types:
            start = back_trace_chars(ann[0], all_text)
            end = forward_trace_chars(ann[1], all_text)
            assert start >= prev_end, "Citation function spans should be disjoint."
            prev_end = end
            citation_mark_span_indices.append((start, end, ann[2]))
    return citation_mark_span_indices


def get_citation_BIO_labels(citation_mark_span_indices, all_text, tokenizer):
    prev_end = 0
    span_labels = []
    for start, end, citation_type in citation_mark_span_indices:
        out_of_span_tokens_len = len(
            tokenizer.tokenize(all_text[prev_end: start]))
        span_labels.extend(["O"] * out_of_span_tokens_len)
        in_span_tokens_len = len(tokenizer.tokenize(all_text[start: end]))
        span_labels.extend(["B_" + citation_type] + ["I_" + citation_type] * (
                    in_span_tokens_len - 1))
        prev_end = end
    out_of_span_tokens_len = len(tokenizer.tokenize(all_text[prev_end:]))
    span_labels.extend(["O"] * out_of_span_tokens_len)
    return ["O"] + span_labels + ["O"]  # CLS and SEP tokens


def get_aligned_BIO_labels(indices, offset_mapping):
    def intersect(start1, end1, start2, end2):
        return end1 > start2 and end2 > start1

    # offset_mapping = tokenizer(all_text, return_offsets_mapping=True)["offset_mapping"]
    pointer = 0
    label = 0
    label_seq = []
    prev_start = -1
    prev_end = -1
    for start, end in offset_mapping[:-1]:
        while pointer < len(indices) and start >= indices[pointer][1]:
            pointer += 1
        if pointer >= len(indices):
            label_seq.append("O")
        elif intersect(start, end, indices[pointer][0], indices[pointer][1]):
            if intersect(prev_start, prev_end, indices[pointer][0],
                         indices[pointer][1]):
                label_seq.append("I_" + indices[pointer][2])
            else:
                label_seq.append("B_" + indices[pointer][2])
        else:
            label_seq.append("O")
        prev_start = start
        prev_end = end
    label_seq.append("O")
    return label_seq


def fix_BIO(label_seqs):
    """
    Some post-processing on the predicted BIO labels to improve the performance a little bit.
    """
    prev_tag = "O"
    new_label_seqs = []
    for label_para in label_seqs:
        new_label_para = []
        for label in label_para:
            if label[:2] == "I_" and prev_tag == "O":
                new_label = "B_" + label[2:]
            elif label[:2] == "B_" and prev_tag[:2] == "B_" and label[
                                                                2:] == prev_tag[
                                                                       2:]:
                new_label = "I_" + label[2:]
            else:
                new_label = label
            prev_tag = new_label
            new_label_para.append(new_label)
        new_label_seqs.append(new_label_para)
    return new_label_seqs


def removeAccents(string):
    # if type(string) is not unicode:
    #    string = unicode(string, encoding='utf-8')

    string = re.sub(u"[àáâãäåȃ]", 'a', string)
    string = re.sub(u"[èéêë]", 'e', string)
    string = re.sub(u"[ìíîï]", 'i', string)
    string = re.sub(u"[òóôõö]", 'o', string)
    string = re.sub(u"[ùúûü]", 'u', string)
    string = re.sub(u"[ýÿỳ]", 'y', string)
    string = re.sub(u"[çćĉč]", 'c', string)
    string = re.sub(u"[šş]", 's', string)
    string = re.sub(u"[ñń]", 'n', string)
    return string


# def removeAccents(string):
#    return str(unicodedata.normalize('NFKD', string).encode('ASCII', 'ignore'))

def citation_prediction_to_annotation_paragraph(dataset, citation_predictions,
                                                tokenizer):
    def post_process(annotations):
        cleaned_annotations = []
        prev_end = 0
        for annotation in annotations:
            start, end = annotation[:2]
            if start >= end - 1:
                continue
            if start >= prev_end:
                cleaned_annotations.append(annotation)
                prev_end = end
            else:
                if cleaned_annotations[-1][1] - cleaned_annotations[-1][
                    1] > end - start:
                    continue
                else:
                    _ = cleaned_annotations.pop()
                    cleaned_annotations.append(annotation)
                    prev_end = end
        return cleaned_annotations

    assert (len(dataset.samples) == len(citation_predictions))
    all_annotations = []
    for sample, citation_prediction in zip(dataset.samples,
                                           citation_predictions):
        # print(sample["id"])
        original_paragraph = sample["paragraph"].replace(tokenizer.sep_token,
                                                         "[BOS]")
        offset_mapping = \
        tokenizer(original_paragraph, return_offsets_mapping=True)[
            "offset_mapping"]
        assert (len(offset_mapping) == len(citation_prediction))

        prev_tag = "O"
        in_span = "O"
        annotations = []
        for i, (mapping, tag) in enumerate(
                zip(offset_mapping, citation_prediction)):
            if i > 0 and i < len(offset_mapping) - 1:
                if tag == "O":
                    in_span = "O"
                elif tag == "B_Dominant":
                    start = mapping[0]
                    in_span = "Dominant"
                elif tag == "B_Reference":
                    start = mapping[0]
                    in_span = "Reference"
                elif in_span != "O" and prev_tag != "O" and tag == "I_" + in_span and \
                        citation_prediction[i + 1] != "I_" + in_span:
                    end = mapping[1]
                    annotations.append(
                        (start, end, in_span, original_paragraph[start: end]))
                prev_tag = tag
            elif i == len(
                    offset_mapping) - 1 and tag != "O" and prev_tag != "O":
                end = offset_mapping[i - 1][1]
                annotations.append(
                    (start, end, prev_tag[2:], original_paragraph[start: end]))

        all_annotations.append(post_process(annotations))
    return all_annotations


def citation_prediction_to_annotation_paragraph_old(dataset,
                                                    citation_predictions,
                                                    tokenizer):
    def post_process(annotations):
        cleaned_annotations = []
        prev_end = 0
        for annotation in annotations:
            start, end = annotation[:2]
            if start >= end - 1:
                continue
            if start >= prev_end:
                cleaned_annotations.append(annotation)
                prev_end = end
            else:
                if cleaned_annotations[-1][1] - cleaned_annotations[-1][
                    1] > end - start:
                    continue
                else:
                    _ = cleaned_annotations.pop()
                    cleaned_annotations.append(annotation)
                    prev_end = end
        return cleaned_annotations

    assert (len(dataset.samples) == len(citation_predictions))
    all_annotations = []
    for sample, citatoin_prediction in zip(dataset.samples,
                                           citation_predictions):
        # print(sample["id"])
        original_paragraph = sample["paragraph"].replace(tokenizer.sep_token,
                                                         "[BOS]")
        tokenized = tokenizer.tokenize(original_paragraph)
        lower_paragraph = removeAccents(original_paragraph.lower())

        end = 0
        prev_token = ""
        prev_tag = "O"
        in_span = "O"
        annotations = []
        start = 0
        for token, tag in zip(tokenized, citatoin_prediction[1:-1]):
            clean_token = token.replace("##", "").lower()
            found_index = lower_paragraph[end:].find(clean_token)
            # assert(found_index >= 0)
            if found_index < 0:
                print(clean_token)
                print(lower_paragraph[end:])
                break
            pointer = found_index + end
            if tag == "B_Dominant":
                start = pointer
                in_span = "Dominant"
            elif tag == "B_Reference":
                start = pointer
                in_span = "Reference"

            # print(token, tag, start, in_span, prev_tag, pointer, end)
            if in_span != "O" and prev_tag != "O" and tag != "I_" + in_span:
                annotations.append(
                    (start, end, in_span, original_paragraph[start: end]))
                in_span = tag[2:] if tag != "O" else tag

            prev_tag = tag
            end = pointer + len(clean_token)
            prev_token = clean_token
        if in_span != "O" and prev_tag[0] == "I":
            annotations.append(
                (start, pointer, in_span, original_paragraph[start: pointer]))
        all_annotations.append(post_process(annotations))
    return all_annotations


def paragraph2doc_annotation(dataset, all_annotations, tokenizer):
    text = ""
    paper_ids = []
    all_texts = []
    annotation_by_doc = []
    for sample, annotations in zip(dataset.samples, all_annotations):
        paper_id, paragraph_id, part_id = sample['id'].split("_")
        if paragraph_id == "0" and part_id == "0":
            paper_ids.append(paper_id)
            if len(text) > 0:
                all_texts.append(text)
                annotation_by_doc.append(this_doc_annotation)
            text = sample["paragraph"].replace(tokenizer.sep_token, "[BOS]")
            this_doc_annotation = annotations

        else:
            offset = len(text)
            text += sample["paragraph"].replace(tokenizer.sep_token, "[BOS]")
            for start, end, label, span in annotations:
                reconstructed_start = start + offset
                reconstructed_end = end + offset
                # print(text[reconstructed_start: reconstructed_end], span)
                # assert(text[reconstructed_start: reconstructed_end] == span)
                if text[reconstructed_start: reconstructed_end] == span:
                    this_doc_annotation.append(
                        (reconstructed_start, reconstructed_end, label, span))

    all_texts.append(text)
    annotation_by_doc.append(this_doc_annotation)
    assert (len(paper_ids) == len(all_texts) == len(annotation_by_doc))
    return paper_ids, all_texts, annotation_by_doc


def span_prediction_to_annotation_paragraph_old(dataset, span_predictions,
                                                tokenizer):
    def post_processing(all_annotations):
        clean_all_annotations = []
        for annotations in all_annotations:
            clean_annotations = []
            if len(annotations) > 0 and annotations[-1][2][0] == "B":
                annotations.pop()
            if len(annotations) % 2 == 0:
                for pair_i in range(len(annotations) // 2):
                    B_start, B_end = annotations[pair_i * 2][:2]
                    E_start, E_end = annotations[pair_i * 2 + 1][:2]
                    if B_start != E_start or B_end != E_end:
                        clean_annotations.append(annotations[pair_i * 2])
                        clean_annotations.append(annotations[pair_i * 2 + 1])
                clean_all_annotations.append(clean_annotations)
            else:
                clean_all_annotations.append(
                    annotations)  # Otherwise just let it go.
        return clean_all_annotations

    assert (len(dataset.samples) == len(span_predictions))
    all_annotations = []
    for sample, span_prediction in zip(dataset.samples, span_predictions):
        # print(sample["id"])
        original_paragraph = sample["paragraph"].replace(tokenizer.sep_token,
                                                         "[BOS]")
        tokenized = tokenizer.tokenize(original_paragraph)
        lower_paragraph = removeAccents(original_paragraph.lower())

        end = 0
        prev_token = ""
        prev_tag = "O"
        in_span = False
        annotations = []
        for token, tag in zip(tokenized, span_prediction[1:-1]):
            clean_token = token.replace("##", "").lower()
            found_index = lower_paragraph[end:].find(clean_token)
            assert (found_index >= 0)
            pointer = found_index + end
            if tag == "B_span":
                in_span = True
                annotations.append((pointer, pointer + len(clean_token),
                                    "B_span", original_paragraph[
                                              pointer: pointer + len(
                                                  clean_token)]))

            # print(token, tag, pointer, in_span, prev_tag)
            if in_span and prev_tag != "O" and tag != "I_span":
                annotations.append((end - len(prev_token), end, "E_span",
                                    original_paragraph[
                                    end - len(prev_token): end]))
                in_span = False if tag == "O" else True
                # print("Appended!")

            prev_tag = tag
            end = pointer + len(clean_token)
            prev_token = clean_token
        if in_span and prev_tag[0] == "I":
            annotations.append(
                (pointer, end, "E_span", original_paragraph[pointer: end]))
        all_annotations.append(annotations)
    return post_processing(all_annotations)


def span_prediction_to_annotation_paragraph(dataset, span_predictions,
                                            tokenizer):
    def post_processing(all_annotations):
        clean_all_annotations = []
        for annotations in all_annotations:
            clean_annotations = []
            if len(annotations) > 0 and annotations[-1][2][0] == "B":
                annotations.pop()
            if len(annotations) % 2 == 0:
                for pair_i in range(len(annotations) // 2):
                    B_start, B_end = annotations[pair_i * 2][:2]
                    E_start, E_end = annotations[pair_i * 2 + 1][:2]
                    if B_start != E_start or B_end != E_end:
                        clean_annotations.append(annotations[pair_i * 2])
                        clean_annotations.append(annotations[pair_i * 2 + 1])
                clean_all_annotations.append(clean_annotations)
            else:
                clean_all_annotations.append(
                    annotations)  # Otherwise just let it go.
        return clean_all_annotations

    assert (len(dataset.samples) == len(span_predictions))
    all_annotations = []
    for sample, span_prediction in zip(dataset.samples, span_predictions):
        original_paragraph = sample["paragraph"].replace(tokenizer.sep_token,
                                                         "[BOS]")
        offset_mapping = \
        tokenizer(original_paragraph, return_offsets_mapping=True)[
            "offset_mapping"]
        assert (len(offset_mapping) == len(span_prediction))

        prev_tag = "O"
        annotations = []
        for i, (mapping, tag) in enumerate(
                zip(offset_mapping, span_prediction)):
            if i > 0 and i < len(offset_mapping) - 1:
                if tag == "B_span":
                    start = mapping[0]
                    end = mapping[1]
                    annotations.append(
                        (start, end, "B_span", original_paragraph[start: end]))

                elif prev_tag != "O" and tag == "I_span" and span_prediction[
                    i + 1] != "I_span":
                    start = mapping[0]
                    end = mapping[1]
                    annotations.append(
                        (start, end, "E_span", original_paragraph[start: end]))
                prev_tag = tag
            elif i == len(
                    offset_mapping) - 1 and tag != "O" and prev_tag != "O":
                start = offset_mapping[i - 1][0]
                end = offset_mapping[i - 1][1]
                annotations.append(
                    (start, end, "E_span", original_paragraph[start: end]))
        all_annotations.append(annotations)
    return post_processing(all_annotations)


def discourse_prediction_to_annotation_paragraph(dataset, discourse_predictions,
                                                 tokenizer):
    assert (len(dataset.samples) == len(discourse_predictions))
    all_annotations = []
    for sample, discourse_prediction in zip(dataset.samples,
                                            discourse_predictions):
        pointer = 0
        annotations = []
        original_paragraph = sample["paragraph"].replace(tokenizer.sep_token,
                                                         "[BOS]")
        for tag in discourse_prediction:
            pointer += original_paragraph[pointer:].find("[BOS]")
            annotations.append((pointer, pointer + len("[BOS]"), tag, "[BOS]"))
            pointer += len("[BOS]")
        all_annotations.append(annotations)
    return all_annotations


# def merge_annotations_by_doc(all_discourses, all_citations, all_spans):
#    merged_annotations = []
#    for discourse, citations, spans in zip(all_discourses, all_citations, all_spans):
#        annotations = discourse + citations + spans
#        this_merged = []
#        for i, annotation in enumerate(annotations):
#            start, end, label, text = annotation
#            this_merged.append(("T"+str(i+1), " ".join([label, str(start), str(end)]), text))
#        merged_annotations.append(this_merged)
#    return merged_annotations

def merge_annotations_by_doc(*all_annotations):
    merged_annotations = []
    N_doc = len(all_annotations[0])
    for annotations in all_annotations:
        assert (len(annotations) == N_doc)

    for i_doc in range(N_doc):
        annotations = []
        for ann in all_annotations:
            annotations += ann[i_doc]
        this_merged = []
        for i, annotation in enumerate(annotations):
            start, end, label, text = annotation
            this_merged.append(("T" + str(i + 1),
                                " ".join([label, str(start), str(end)]), text))
        merged_annotations.append(this_merged)
    return merged_annotations


def write_brat(path, paper_ids, all_texts, merged_annotations):
    if not os.path.exists(path):
        os.makedirs(path)
    for paper_id, all_text, merged_annotation in zip(paper_ids, all_texts,
                                                     merged_annotations):
        with open(os.path.join(path, paper_id + ".txt"), "w") as wf:
            wf.write(all_text)
        with open(os.path.join(path, paper_id + ".ann"), "w") as f:
            for annotation in merged_annotation:
                f.write("\t".join(annotation) + "\n")
    print("Done writing!")


def discourse2bos(augmented_paragraph, discourse_tokens):
    for token in discourse_tokens:
        augmented_paragraph = augmented_paragraph.replace(token, "[BOS]")
    return augmented_paragraph


def get_citation_type(sentence, found_pointer):
    def find_B(match):
        for token in reversed(match):
            if token in {"[B_Dominant]", "[B_Reference]"}:
                return token[3:-1]
            elif token in {"[I_Dominant]", "[I_Reference]"}:
                return ""
        return ""

    def find_E(match):
        for token in match:
            if token in {"[B_Dominant]", "[B_Reference]"}:
                return ""
            elif token in {"[E_Dominant]", "[E_Reference]"}:
                return token[3:-1]
        return ""

    match_before = re.findall('\[[a-zA-Z_]*\]', sentence[:found_pointer])
    match_after = re.findall('\[[a-zA-Z_]*\]', sentence[found_pointer:])
    B = find_B(match_before)
    E = find_E(match_after)
    if len(B) > 0 and B == E:
        return B
    else:
        return ""


def read_related_work_jsons(related_work_path):
    related_work_jsons = {}
    with open(related_work_path, "r") as f_pdf:
        for line in f_pdf:
            related_work_dict = json.loads(line)
            related_work_jsons[
                related_work_dict["paper_id"]] = related_work_dict
    return related_work_jsons


def make_augmented_paragraphs(tokens, tokenizer, discourse_tokens=None,
                              discourse_labels=None, span_BIO_labels=None,
                              citation_BIO_labels=None):
    augmented_text = ""
    prev_span_BIO_label = "O"
    prev_citation_BIO_label = "O"
    in_citation = ""
    discourse_pointer = 0
    in_span = False
    for i in range(len(tokens)):
        if span_BIO_labels is not None and span_BIO_labels[i] == "B_span":
            if prev_span_BIO_label == "I_span":
                augmented_text += "[E_span] "
            augmented_text += "[B_span] "
            in_span = True
        if citation_BIO_labels is not None and citation_BIO_labels[i][
                                               :2] == "B_":
            if prev_citation_BIO_label[:2] == "I_":
                augmented_text += "[E_" + in_citation + "] "
            augmented_text += "[" + citation_BIO_labels[i] + "] "
            in_citation = citation_BIO_labels[i][2:]
        if citation_BIO_labels is not None and citation_BIO_labels[
            i] == "O" and prev_citation_BIO_label != "O":
            augmented_text += "[E_" + in_citation + "] "
            in_citation = ""
        if span_BIO_labels is not None and span_BIO_labels[
            i] == "O" and prev_span_BIO_label != "O":
            augmented_text += "[E_span] "
            in_span = False
        if discourse_labels is not None and tokens[i] == "[BOS]":
            augmented_text += "[" + discourse_labels[discourse_pointer] + "] "
            discourse_pointer += 1
        else:
            augmented_text += tokens[i] + " "
        if span_BIO_labels is not None:
            prev_span_BIO_label = span_BIO_labels[i]
        if citation_BIO_labels is not None:
            prev_citation_BIO_label = citation_BIO_labels[i]
    if len(in_citation) > 0:
        augmented_text += "[E_" + in_citation + "] "
    if in_span:
        augmented_text += "[E_span]"
    augmented_paragraph = tokenizer.convert_tokens_to_string(
        augmented_text.split())
    if discourse_tokens is not None:
        augmented_paragraph_bos = discourse2bos(augmented_paragraph,
                                                discourse_tokens)
        augmented_sentences = augmented_paragraph_bos.split("[BOS] ")[1:]
    else:
        augmented_sentences = augmented_paragraph.split("[BOS] ")[1:]
    return augmented_paragraph, augmented_sentences


def sentence_citation_link(paragraph_id, augmented_sentences,
                           related_work_jsons, tokenizer):
    paper_id, p_id, part_id = paragraph_id.split("_")
    paragraph_citation_links = []
    for sentence in augmented_sentences:
        this_sentence_citations = {'Dominant': {}, 'Reference': {}}
        for citation in related_work_jsons[paper_id]["related_work"][int(p_id)][
            "cite_spans"]:
            tokenized_citation_mark = tokenizer.convert_tokens_to_string(
                tokenizer.tokenize(citation["text"]))
            found_pointer = sentence.find(tokenized_citation_mark)
            if found_pointer >= 0:
                key = get_citation_type(sentence, found_pointer)
                if len(key) > 0:
                    link = related_work_jsons[paper_id]["bib_entries"][
                        citation["ref_id"]]["link"]
                    this_sentence_citations[key][tokenized_citation_mark] = link
                else:
                    print(paragraph_id)
                    print(sentence)
                    # print(tokenized_citation_mark)
        paragraph_citation_links.append(this_sentence_citations)
    return paragraph_citation_links


def span_sentence_map(augmented_sentences):
    # Identify spans cross sentences.
    span_sent_mapping = [[] for i in range(len(augmented_sentences))]
    i_span = 0
    carry_on = False
    for si, sentence in enumerate(augmented_sentences):
        if carry_on:
            span_sent_mapping[si].append(i_span)
        for token in sentence.split():
            if token == "[B_span]":
                span_sent_mapping[si].append(i_span)
                carry_on = True
            elif token == "[E_span]":
                carry_on = False
                i_span += 1
    return span_sent_mapping, i_span


def propagate_citation_cross_sentences(span_sent_mapping,
                                       paragraph_citation_links, i_span):
    # Propagate citations within spans.
    for span_idx in range(i_span):
        dominant_dict = {}
        reference_dict = {}
        for si, sent in enumerate(span_sent_mapping):
            if span_idx in sent:
                dominant_dict.update(paragraph_citation_links[si]["Dominant"])
                reference_dict.update(paragraph_citation_links[si]["Reference"])
        for si, sent in enumerate(span_sent_mapping):
            if span_idx in sent:
                paragraph_citation_links[si]["Dominant"] = dominant_dict
                paragraph_citation_links[si]["Reference"] = reference_dict
    return paragraph_citation_links


def mount_discourse_sentence(discourse_labels, sentences):
    return " ".join(
        ["[" + d + "] " + s for d, s in zip(discourse_labels, sentences)])


def find_span(paragraph, tokenizer, begin_token, end_token, truncate=False):
    span_indices = []
    start = 0
    stack_size = 0
    tokens = tokenizer.tokenize(paragraph)
    for i, token in enumerate(tokens):
        if begin_token in token:
            stack_size += 1
            if stack_size == 1:
                start = i
        if end_token in token:
            stack_size -= 1
            if stack_size == 0:
                span_indices.append((start, i + 1))
    out = []
    for pair in span_indices:
        if truncate and pair[1] - pair[0] > 8:
            out.extend(tokens[pair[0]: pair[0] + 4] + ["$"] + tokens[
                                                              pair[1] - 4: pair[
                                                                  1]])
        else:
            out.extend(tokens[pair[0]: pair[1]])
    return tokenizer.convert_tokens_to_string(out)


def makeMaskedLanguageModelSample(paragraph, tokenizer):
    tokens = []
    for token in tokenizer.tokenize(paragraph):
        if token[:2] == "##":
            tokens[-1] += token[2:]
        else:
            tokens.append(token)

    i = 0
    noisy_inputs = "MLM: "
    targets = []
    while i < len(tokens):
        if random.random() < 0.05:
            noisy_inputs += "[MASK] "
            span = random.choice([1, 2, 3, 4, 5])
            targets.append(" [SEP] " + " ".join(tokens[i:i + span]))
            i += span
        else:
            noisy_inputs += tokens[i] + " "
            i += 1
    target_string = ("".join(targets) + " [SEP]").strip()
    return noisy_inputs, target_string


def makeSentenceReorderingSample(paragraph):
    sentences = sent_tokenize(paragraph)
    indices = [i for i in range(len(sentences))]
    random.shuffle(indices)
    shuffled_sentences = "Reorder:"
    for index in indices:
        shuffled_sentences += " [BOS] " + sentences[index]
    shuffled_sentences = shuffled_sentences.strip()
    target_indices = " ".join([str(i) for i in indices])
    return shuffled_sentences, target_indices


def normalize_section(section):
    section = section.lower()
    mapping = {
        "title": "title",
        "abstract": "abstract",
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
    for k, v in mapping.items():
        if k in section:
            return v
    return "other"


def post_process_spans(span_predictions, citation_predictions):
    """
    Enforce the rule that each span must contain at least one citation.
    Eliminate false-positive span predictions.
    """

    new_span_predictions = []
    for span_prediction, citation_prediction in zip(span_predictions,
                                                    citation_predictions):
        assert (len(span_prediction) == len(citation_prediction))
        span_ranges = []
        start = 0
        for i, span_pred in enumerate(span_prediction):
            if span_pred == "B_span":
                start = i
            elif span_pred == "I_span":
                if (i + 1 < len(span_prediction) and span_prediction[
                    i + 1] == "O") or (i == len(span_prediction) - 1):
                    span_ranges.append((start, i + 1))

        prev_end = 0
        new_span_prediction = []
        for start, end in span_ranges:
            new_span_prediction.extend(span_prediction[prev_end:start])
            if len(set(citation_prediction[start:end]).difference({"O"})) > 0:
                new_span_prediction.extend(span_prediction[start:end])
            else:
                new_span_prediction.extend(["O"] * (end - start))
            prev_end = end
        new_span_prediction.extend(span_prediction[prev_end:])
        new_span_predictions.append(new_span_prediction)
    return new_span_predictions


def discourse_tag2seq(paragraph, labels, tokenizer):
    seq_labels = []
    for token in tokenizer.tokenize(paragraph):
        if token == "[BOS]":
            seq_labels.append(labels.pop(0))
        else:
            seq_labels.append("O")
    assert len(labels) == 0
    return seq_labels


def custom_span_token2T5(span_label):
    placeholder = "[TOKEN]"
    seq = span_label.replace("[E_span] [B_span]", placeholder)
    seq = seq.replace("[B_span]", placeholder)
    seq = seq.replace("[E_span]", placeholder)

    count = 0
    new_seq = []
    for token in seq.split():
        if token == placeholder:
            new_seq.append("<extra_id_" + str(count) + ">")
            count += 1
        else:
            new_seq.append(token)
    return " ".join(new_seq)


def custom_citation_token2T5(dominant_label, reference_label):
    placeholder = "[TOKEN]"
    dominant_seq = dominant_label.replace("[E_Dominant] [B_Dominant]",
                                          placeholder)
    dominant_seq = dominant_seq.replace("[B_Dominant]", placeholder)
    dominant_seq = dominant_seq.replace("[E_Dominant]", placeholder)

    reference_seq = reference_label.replace("[E_Reference] [B_Reference]",
                                            placeholder)
    reference_seq = reference_seq.replace("[B_Reference]", placeholder)
    reference_seq = reference_seq.replace("[E_Reference]", placeholder)

    count = 0
    new_seq = ["Dominant:"]
    for token in dominant_seq.split():
        if token == placeholder:
            new_seq.append("<extra_id_" + str(count) + ">")
            count += 1
        else:
            new_seq.append(token)

    new_seq.append("Reference:")
    for token in reference_seq.split():
        if token == placeholder:
            new_seq.append("<extra_id_" + str(count) + ">")
            count += 1
        else:
            new_seq.append(token)
    return " ".join(new_seq)


def makeMLMsample(text, mask_token="<mask>"):
    tokens = word_tokenize(text)

    i = 0
    noisy_inputs = ""
    targets = []
    while i < len(tokens):
        if random.random() < 0.05:
            noisy_inputs += mask_token + " "
            i += random.choice([1, 2, 3, 4, 5])
        else:
            noisy_inputs += tokens[i] + " "
            i += 1
    return noisy_inputs, " ".join(tokens)


def s2orc_to_corwa_paragraph_index(paragraph_id, sentences, related_work_jsons,
                                   offset_mapping, citation_labels,
                                   separator="[BOS] "):
    # This implementation is relying on the extracted citation marks and links from S2ORC, while the citation extraction from S2ORC is not complete. There are many citations missed by S2ORC.
    def get_citation_type(citation_label_seq):
        counts = {"Dominant": 0, "Reference": 0}
        for label in citation_label_seq:
            if len(label) > 2:
                counts[label[2:]] += 1
        return "Dominant" if counts["Dominant"] >= counts[
            "Reference"] else "Reference"

    paper_id, p_id, part_id = paragraph_id.split("_")
    paragraph_citation_links = [{'Dominant': {}, 'Reference': {}} for sent in
                                sentences]
    bib_entries = related_work_jsons[paper_id]["bib_entries"]
    citations = related_work_jsons[paper_id]["related_work"][int(p_id)][
        "cite_spans"]

    # "".join(sentences) == original_s2orc_paragraph    
    offset = len(separator)
    i_sentence = 0
    i_token = 0
    charactor_pointer = 0
    informative_citations = []
    for citation in citations:
        this_citation = copy.copy(citation)
        while this_citation["start"] >= charactor_pointer + len(
                sentences[i_sentence]):
            charactor_pointer += len(sentences[i_sentence])
            i_sentence += 1
        this_citation["i_sentence"] = i_sentence
        this_citation["corwa_start"] = this_citation["start"] + (
                    i_sentence + 1) * len(separator)
        this_citation["corwa_end"] = this_citation["end"] + (
                    i_sentence + 1) * len(separator)

        in_span_token_ids = []
        while offset_mapping[i_token][1] <= this_citation["corwa_start"]:
            i_token += 1
        while offset_mapping[i_token][0] < this_citation["corwa_end"] and \
                offset_mapping[i_token][1] > 0:
            in_span_token_ids.append(i_token)
            # print(len(offset_mapping), i_token, offset_mapping[i_token], this_citation["corwa_end"])
            i_token += 1

        this_citation["in_span_token_ids"] = in_span_token_ids
        citation_label_seq = citation_labels[
                             in_span_token_ids[0]: in_span_token_ids[-1] + 1]
        this_citation["citation_type"] = get_citation_type(citation_label_seq)
        this_citation["link"] = bib_entries[this_citation["ref_id"]]["link"]

        informative_citations.append(this_citation)
    return informative_citations


def new_sentence_citation_link(citations, N_sent):
    paragraph_citation_links = [{'Dominant': {}, 'Reference': {}} for i in
                                range(N_sent)]
    for citation in citations:
        paragraph_citation_links[citation["i_sentence"]][
            citation["citation_type"]][citation["text"]] = citation["link"]
    return paragraph_citation_links


def new_span_sentence_map(tokens, span_BIO_labels, bos="[BOS]"):
    i_span = 0
    carry_on = False
    span_sent_mapping = []
    current_sentence = []
    prev_label = "O"
    for i, (token, label) in enumerate(zip(tokens, span_BIO_labels)):
        if prev_label == "I_span" and label != "I_span":
            carry_on = False
            i_span += 1

        if token == bos:
            span_sent_mapping.append([])
            if carry_on:
                span_sent_mapping[-1].append(i_span)
            # print(span_sent_mapping)

        if label == "B_span":
            span_sent_mapping[-1].append(i_span)
            # print(span_sent_mapping)
            carry_on = True

        prev_label = label
    return span_sent_mapping, i_span

def previous_sentence_end_index_from_span(paragraph:str, char_start:int):
    """
    get previous sentence ending index based on span start index
    """
    i = 0
    
    for line in paragraph.splitlines():
        end = i + len(line)
        if i <= char_start <= end:
            return i - 1
        i = end + 1
    return i


def next_sentence_start_index_from_span(paragraph:str, char_start:int):
    """
    get next sentence starting index based on span start index
    """
    i = 0

    for line in paragraph.splitlines():
        end = i + len(line)
        if i <= char_start <= end:
            return end + 1
        i = end + 1
    return i


def map_span_citation_sentence(span_labels, citation_labels, pargraph_citation_info, offset_mapping):
    span_ranges, span_types = span_range_token_indices(span_labels, citation_labels)
    span_citation_mapping = []
    span_char_ranges = []
    for span_range in span_ranges:
        this_span = {'Dominant': {}, 'Reference': {}}
        for citation in pargraph_citation_info:
            if citation["in_span_token_ids"][0] >= span_range[0] and \
                    citation["in_span_token_ids"][-1] < span_range[1]:
                this_span[citation["citation_type"]][citation["text"]] = \
                citation["link"]
                this_span[citation["citation_type"]]["{}_pos".format(citation["text"])] = \
                    (citation["corwa_start"], citation["corwa_end"])
        span_citation_mapping.append(this_span)
        span_char_ranges.append((offset_mapping[span_range[0]][0],
                                 offset_mapping[span_range[-1] - 1][-1]))

    span_citation_mapping_info = []
    for span_range, span_type, mapping, char_range in zip(span_ranges,
                                                span_types,
                                               span_citation_mapping,
                                               span_char_ranges):
        info_dict = {"token_start": span_range[0], "token_end": span_range[-1],
                     "char_start": char_range[0], "char_end": char_range[-1],
                     "span_type": span_type,
                     "span_citation_mapping": mapping}
        span_citation_mapping_info.append(info_dict)

    return span_citation_mapping_info


def map_span_citation_old(span_labels, pargraph_citation_info, offset_mapping):
    span_ranges = span_range_token_indices_old(span_labels)
    span_citation_mapping = []
    span_char_ranges = []
    for span_range in span_ranges:
        this_span = {'Dominant': {}, 'Reference': {}}
        for citation in pargraph_citation_info:
            if citation["in_span_token_ids"][0] >= span_range[0] and \
                    citation["in_span_token_ids"][-1] < span_range[1]:
                this_span[citation["citation_type"]][citation["text"]] = \
                citation["link"]
        span_citation_mapping.append(this_span)
        span_char_ranges.append((offset_mapping[span_range[0]][0],
                                 offset_mapping[span_range[-1] - 1][-1]))

    span_citation_mapping_info = []
    for span_range, mapping, char_range in zip(span_ranges,
                                               span_citation_mapping,
                                               span_char_ranges):
        info_dict = {"token_start": span_range[0], "token_end": span_range[-1],
                     "char_start": char_range[0], "char_end": char_range[-1],
                     "span_citation_mapping": mapping}
        span_citation_mapping_info.append(info_dict)

    return span_citation_mapping_info


def map_span_citation(span_labels, citation_labels, pargraph_citation_info, offset_mapping):
    span_ranges, span_types = span_range_token_indices(span_labels, citation_labels)
    span_citation_mapping = []
    span_char_ranges = []
    for span_range in span_ranges:
        this_span = {'Dominant': {}, 'Reference': {}}
        for citation in pargraph_citation_info:
            if citation["in_span_token_ids"][0] >= span_range[0] and \
                    citation["in_span_token_ids"][-1] < span_range[1]:
                this_span[citation["citation_type"]][citation["text"]] = \
                citation["link"]
        span_citation_mapping.append(this_span)
        span_char_ranges.append((offset_mapping[span_range[0]][0],
                                 offset_mapping[span_range[-1] - 1][-1]))

    span_citation_mapping_info = []
    for span_range, span_type, mapping, char_range in zip(span_ranges,
                                               span_types,
                                               span_citation_mapping,
                                               span_char_ranges):
        info_dict = {"token_start": span_range[0], "token_end": span_range[-1],
                     "char_start": char_range[0], "char_end": char_range[-1],
                     "span_type": span_type,
                     "span_citation_mapping": mapping}
        span_citation_mapping_info.append(info_dict)

    return span_citation_mapping_info


def span_range_token_indices_old(span_labels):
    prev_label = "O"
    span_ranges = []
    for i, label in enumerate(span_labels):
        if prev_label == "I_span" and label != "I_span":
            end = i
            span_ranges.append((start, end))
        if label == "B_span":
            start = i
        prev_label = label
    return span_ranges

def span_range_token_indices(span_labels, citation_labels):
    prev_label = "O"
    span_ranges = []
    span_types = []
    for i, label in enumerate(span_labels):
        if prev_label == "I_span" and label != "I_span":
            end = i
            span_ranges.append((start, end))
            citation_types = set(citation_labels[start: end])
            if "B_Dominant" in citation_types or "I_Dominant" in citation_types:
                span_types.append("Dominant")
            else:
                span_types.append("Reference")
        if label == "B_span":
            start = i
        prev_label = label
    return span_ranges, span_types


def restore_citation_mark(citation_mark):
    citation_mark = citation_mark.replace("( ", "(")
    citation_mark = citation_mark.replace(" )", ")")
    citation_mark = citation_mark.replace("[ ", "[")
    citation_mark = citation_mark.replace(" ]", "]")
    citation_mark = citation_mark.replace("et al ", "et al. ")
    citation_mark = citation_mark.replace(" ,", ",")
    return citation_mark


def citation_by_sentence(tokens, citation_BIO_labels):
    dominant_count = 0
    reference_count = 0
    counts = []
    for i in range(len(tokens)):
        if citation_BIO_labels[i] == "B_Dominant":
            dominant_count += 1
        elif citation_BIO_labels[i] == "B_Reference":
            reference_count += 1
        if tokens[i] == "[BOS]":
            if i > 0:
                counts.append((dominant_count, reference_count))
            dominant_count = 0
            reference_count = 0
    counts.append((dominant_count, reference_count))
    return counts


def scires_function_by_paper_id(predictions, dev_set, task):
    prediction_set = []
    for pred, data in zip(predictions, dev_set.samples):
        data["{}_label".format(task)] = pred
        prediction_set.append(data)

    paper_ids = set([x["paper_id"] for x in dev_set.samples])
    paper_mapped_data = defaultdict(list)
    for paper_id in paper_ids:

        for data in prediction_set:
            if data["paper_id"] == paper_id:
                paper_mapped_data[paper_id].append(data)
    return paper_mapped_data

def makeBertMLMsample(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    noisy_inputs = []
    mask_count = 0
    for token in tokens:
        if token[:2] == "##":
            if mask_count > 0:
                noisy_inputs.append(tokenizer.mask_token)
            else:
                noisy_inputs.append(token)
        elif mask_count == 0 and random.random() < 0.05:
            mask_count = random.choice([1, 2, 3, 4, 5])
            noisy_inputs.append(tokenizer.mask_token)
            mask_count -= 1
        elif mask_count > 0:
            noisy_inputs.append(tokenizer.mask_token)
            mask_count -= 1
        else:
            noisy_inputs.append(token)
    return tokenizer.convert_tokens_to_string(noisy_inputs), tokenizer.convert_tokens_to_string(tokens)


def crossvalid(dataset=None, k_fold=5):
    total_size = len(dataset)
    fraction = 1 / k_fold
    seg = int(total_size * fraction)

    for i in range(k_fold):
        trll, trlr = 0, i * seg
        vall, valr = trlr, i * seg + seg
        trrl, trrr = valr, total_size

        train_left_indices = list(range(trll, trlr))
        train_right_indices = list(range(trrl, trrr))

        train_indices = train_left_indices + train_right_indices
        val_indices = list(range(vall, valr))

        train_set = [dataset[i] for i in train_indices]
        val_set = [dataset[i] for i in val_indices]
        yield train_set, val_set


# adapted from fairseq utils
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

def make_positions(tensor, padding_idx, left_pad, onnx_trace=False):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """
    if onnx_trace:
        range_buf = torch._dim_arange(like=tensor, dim=1) + padding_idx + 1
        mask = tensor.ne(padding_idx)
        positions = range_buf.expand_as(tensor)
        if left_pad:
            positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
        return positions * mask.long() + positions * (1 - mask.long())

    max_pos = padding_idx + 1 + tensor.size(1)
    if not hasattr(make_positions, 'range_buf'):
        make_positions.range_buf = tensor.new()
    make_positions.range_buf = make_positions.range_buf.type_as(tensor)
    if make_positions.range_buf.numel() < max_pos:
        torch.arange(padding_idx + 1, max_pos, out=make_positions.range_buf)
    mask = tensor.ne(padding_idx)
    positions = make_positions.range_buf[:tensor.size(1)].expand_as(tensor)
    if left_pad:
        positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
    return tensor.clone().masked_scatter_(mask, positions[mask])

def padding_tensor(sequences, max_len=None, pad_value=-100):
    """
    :param sequences: list of tensors
    :return:
    """
    num = len(sequences)
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    out_dims = (num, max_len)
    out_tensor = sequences[0].data.new(*out_dims).fill_(pad_value)
    mask = sequences[0].data.new(*out_dims).fill_(0)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        if length > max_len:
            # when length is bigger than max_len, shrink the tensor
            out_tensor[i, :max_len] = tensor[:max_len]
            mask[i, :max_len] = 1

        else:
            out_tensor[i, :length] = tensor
            mask[i, :length] = 1
    return out_tensor, mask


def reverse_arange(x):
    return torch.flip(torch.arange(start=1, end = x.cpu().numpy().item() + 1), dims=(0,) )

def get_length_positions_and_segment_ids(
    axis: int = 0,
    context_length_prev=None,
    span_length=None,
    context_length_next=None,
    max_len = None
):
    len_positions, segment_ids = [], []
    for len_prev, len_span, len_next in zip(
        torch.unbind(context_length_prev, axis=axis),
        torch.unbind(span_length, axis=axis),
        torch.unbind(context_length_next, axis=axis)
    ):  
        prev, span, nex = reverse_arange(len_prev), reverse_arange(len_span), reverse_arange(len_next)

        res =  torch.cat([prev, span, nex])
        
        segment_id = torch.cat( [torch.ones_like(prev, dtype=torch.int64) * 0,
        torch.ones_like(span, dtype=torch.int64) * 1,
        torch.ones_like(nex, dtype=torch.int64) * 2] )
        
        len_positions.append(res)
        segment_ids.append(segment_id)
    return padding_tensor(len_positions, max_len=max_len, pad_value=0), padding_tensor(segment_ids, max_len=max_len, pad_value=2)


def get_scaler_val(tensor):
    return tensor.cpu().detach().numpy().item()

