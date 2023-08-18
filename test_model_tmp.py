import json
import jsonlines
import random
import re
import string
import sys
import collections
from collections import Counter

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from qa.reader_model import Reader
from retrieval.retriever_model import Retriever
from tqdm import tqdm
import argparse
import numpy as np

def load_saved(model, path, exact=True):
    try:
        state_dict = torch.load(path)
    except:
        state_dict = torch.load(path, map_location=torch.device('cpu'))

    def filter(x):
        return x[7:] if x.startswith('module.') else x

    if exact:
        state_dict = {filter(k): v for (k, v) in state_dict.items()}
    else:
        state_dict = {filter(k): v for (k, v) in state_dict.items() if filter(k) in model.state_dict()}
    model.load_state_dict(state_dict)
    return model

def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda()
        elif isinstance(maybe_tensor, dict):
            return {
                key: _move_to_cuda(value)
                for key, value in maybe_tensor.items()
            }
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_cuda(sample)

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def calculate_em_f1(predicted_support_idxs, gold_support_idxs):
    # Taken from hotpot_eval
    cur_sp_pred = set(map(int, predicted_support_idxs))
    gold_sp_pred = set(map(int, gold_support_idxs))
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0

    # In case everything is empty, set both f1, em to be 1.0.
    # Without this change, em gets 1 and f1 gets 0
    if not cur_sp_pred and not gold_sp_pred:
        f1, em = 1.0, 1.0
        f1, em = 1.0, 1.0
    return f1, em

def normalize_sp(sps):
    new_sps = []
    for sp in sps:
        sp = list(sp)
        sp[0] = sp[0].lower()
        new_sps.append(sp)
    return new_sps


def update_sp(prediction, gold):
    cur_sp_pred = normalize_sp(set(map(tuple, prediction)))
    gold_sp_pred = normalize_sp(set(map(tuple, gold)))
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    return em, f1

def get_retr_output(test_raw_data, type='musique', is_dev=True, beam_size=1):
    retr_dic = {}
    if type == '2wiki':
        re_tokenizer_path = 'model/deberta-v3-base'
    else:
        re_tokenizer_path = 'model/deberta-v3-large'
    re_model_path = re_tokenizer_path
    if type == 'musique':
        re_checkpoint = 'project/hotpotqa/retr_beamsize2_793.pt'
    else:
        re_checkpoint = 'project/hotpotqa/2407_codes/output/07-24-2023/train_2wiki_continue_training-seed42-bsz8-fp16True-lr1e-05-decay0.0-warm0.1-valbsz1/checkpoint_best.pt'
    pred_filename = f"pred_test_{type}_v0_retr.json"
    max_len = 512
    mean_passage_len = 250 if type=='hotpot' else 120 
    device = torch.device("cuda", 1)
    tokenizer = AutoTokenizer.from_pretrained(re_tokenizer_path)
    config = AutoConfig.from_pretrained(re_tokenizer_path)
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    re_model = Retriever(config, re_model_path, encoder_class=AutoModel, mean_passage_len=mean_passage_len, beam_size=beam_size, gradient_checkpointing=True)
    re_model = load_saved(re_model, re_checkpoint)
    re_model = re_model.to(device)
    re_model.eval()
    if is_dev:
        em_tot, f1_tot = [], []
    # get tensors
    for sample in tqdm(test_raw_data, desc="RE Predicting:"):
        question = sample['question']
        if question.endswith("?"):
            question = question[:-1]
        id = sample['id'] if type == 'musique' else sample['_id']
        q_codes = tokenizer.encode(question, add_special_tokens=False, return_tensors="pt", truncation=True, max_length=max_len).squeeze(0)
        c_codes = []
        if is_dev:
            sf_idx = []
            sp_title_set = set()
        if type == 'hotpot' or type == '2wiki':
            for idx, (title, sentences) in enumerate(sample['context']):
                if is_dev:
                    for sup in sample['supporting_facts']:
                        sp_title_set.add(sup[0])
                    if title in sp_title_set:
                        sf_idx.append(idx)
                l = title + "".join(sentences)
                encoding = tokenizer.encode(l, add_special_tokens=False, return_tensors="pt", truncation=True, max_length=max_len-q_codes.shape[-1]).squeeze(0)
                encoding = encoding.to(device)
                c_codes.append(encoding)
        elif type == 'musique':
            # musique
            for i, para in enumerate(sample['paragraphs']):
                if is_dev:
                    if para['is_supporting']:
                        sf_idx.append(i)
                l = para['title'] + '.' + para['paragraph_text']
                encoding = tokenizer.encode(l, add_special_tokens=False, return_tensors="pt", truncation=True, max_length=max_len-q_codes.shape[-1]).squeeze(0)
                encoding = encoding.to(device)
                c_codes.append(encoding)
        q_codes = q_codes.to(device)
        q_codes_input = [q_codes]
        c_codes_input = [c_codes]
        hop = int(id[0]) if type == 'musique' else 2
        if type == '2wiki' and sample['type'] == 'bridge_comparison':
            hop = 4
        with torch.no_grad():
            current_preds = re_model(q_codes_input, c_codes_input, [] if not is_dev else sf_idx, hop=hop)['current_preds']
        retr_dic[id] = current_preds[0]
        if is_dev:
            f1, em = calculate_em_f1(current_preds[0], sf_idx)
            em_tot.append(em)
            f1_tot.append(f1)
    if is_dev:
        print(f"em:{sum(em_tot) / len(em_tot)}, f1:{sum(f1_tot) / len(f1_tot)}")
    with open(pred_filename, "w", encoding="utf-8") as f:
        json.dump(retr_dic, f, ensure_ascii=False, indent=4)
    print(f"retr evaluation finished!")
    torch.cuda.empty_cache()
    return retr_dic

def merge_find_ans(start_logits, end_logits, ids, punc_token_list, topk=5, max_ans_len=20):
    def is_too_long(span_id, punc_token_list):
        for punc_token_id in punc_token_list:
            if punc_token_id in span_id:
                return True
        return False
    start_candidate_val, start_candidate_idx = start_logits.topk(topk, dim=-1)
    end_candidate_val, end_candidate_idx = end_logits.topk(topk, dim=-1)
    pointer_s, pointer_e = 0, 0
    start = start_candidate_idx[pointer_s].item()
    end = end_candidate_idx[pointer_e].item()
    span_id = ids[start: end + 1]
    while start > end or (end - start) > max_ans_len or is_too_long(span_id, punc_token_list):
        if start_candidate_val[pointer_s] > end_candidate_val[pointer_e]:
            pointer_e += 1
        else:
            pointer_s += 1
        if pointer_s >= topk or pointer_e >= topk:
            break
        start = start_candidate_idx[pointer_s].item()
        end = end_candidate_idx[pointer_e].item()
        span_id = ids[start: end + 1]
    return span_id

def get_reader_qa_output(retr_pred_dic, test_raw_data, type='musique', is_dev=True, answer_merge=False, topk=5):
    qa_tokenizer_path = "model/deberta-v3-large-squad2"
    qa_model_path = qa_tokenizer_path
    if type == '2wiki':
        qa_checkpoint = 'project/hotpotqa/2107_codes/output/07-21-2023/2wiki_multi_reader_large-seed42-bsz4-fp16True-lr1e-05-decay0.0-warm0.1-valbsz32/checkpoint_best.pt'
    else:
        qa_checkpoint = "project/hotpotqa/3107_codes/output/08-01-2023/musique_reader_deberta_large_from_scratch-seed42-bsz8-fp16True-lr6e-06-decay0.0-warm0.1-valbsz32/checkpoint_best.pt"
    pred_filename = f"sorted_pred_{'dev' if is_dev else 'test'}_{type}_v0_retrlarge_793_qalarge_70_{'merged' if answer_merge else 'no_merged'}.{'jsonl' if type=='musique' else 'json'}"
    max_len = 1024
    device = torch.device("cuda", 1)
    config = AutoConfig.from_pretrained(qa_model_path)
    config.max_position_embeddings = max_len
    tokenizer = AutoTokenizer.from_pretrained(qa_tokenizer_path)
    type = 'hotpot' if type == '2wiki' else type
    if type == 'hotpot':
        SEP = "</e>"
        DOC = "</d>"
        tokenizer.add_tokens([SEP, DOC])
        SEP_id = tokenizer.convert_tokens_to_ids(SEP)
        DOC_id = tokenizer.convert_tokens_to_ids(DOC)
        sp_pred = {}
        ans_pred = {}
    qa_model = Reader(config, qa_model_path, len(tokenizer) if ('deberta' not in qa_tokenizer_path) else 0)
    qa_model = load_saved(qa_model, qa_checkpoint)
    qa_model = qa_model.to(device)
    qa_model.eval()
    pred_list = []
    if is_dev:
        em_tot, f1_tot = [], []
        if type == 'hotpot':
            sp_em_tot, sp_f1_tot = [], []
    # get tensors
    for sample in tqdm(test_raw_data, desc="QA Predicting:"):
        question = sample['question']
        if question.endswith("?"):
            question = question[:-1]
        id = sample['id'] if type == 'musique' else sample['_id']
        q_codes = tokenizer.encode(question, add_special_tokens=False, truncation=True, max_length=max_len)
        sp_list = retr_pred_dic[id]
        idx2title = {}
        c_codes = []
        if type == 'hotpot':
            # hotpot format
            sts2title = {}
            sts2idx = {}
            sts_idx = 0
            sentence_label = []
            if is_dev:
                sp_title_set = {}
                for sup in sample['supporting_facts']:
                    if sup[0] not in sp_title_set:
                        sp_title_set[sup[0]] = []
                    sp_title_set[sup[0]].append(sup[1])
            for idx, (title, sentences) in enumerate(sample['context']):
                if idx in sp_list:
                    idx2title[idx] = title
                    l = DOC + " " + title
                    for idx2, c in enumerate(sentences):
                        l += (SEP + " " + c)
                        # sts2title[sts_idx] = title
                        # sts2idx[sts_idx] = idx2
                        # if is_dev:
                        #     if title in sp_title_set and idx2 in sp_title_set[title]:
                        #         sentence_label.append(1)
                        #     else:
                        #         sentence_label.append(0)
                        # sts_idx += 1
                    encoding = tokenizer.encode(l, add_special_tokens=False, truncation=True, max_length=max_len-len(q_codes))
                    c_codes.append(encoding)
        elif type == 'musique':
            # musique
            for i, para in enumerate(sample['paragraphs']):
                if i in sp_list:
                    l = para['title'] + '.' + para['paragraph_text']
                    encoding = tokenizer.encode(l, add_special_tokens=False, truncation=True, max_length=max_len-len(q_codes))
                    c_codes.append(encoding)
        total_len = len(q_codes) + sum([len(item) for item in c_codes])
        context_ids = [tokenizer.cls_token_id] + q_codes
        avg_len = (max_len - 2 - len(q_codes)) // len(c_codes)
        
        if type == 'hotpot':
            sp_list.sort() # only hotpot format, for sp prediction, align sentence order and passages order
        for idx, item in enumerate(c_codes):
            if total_len > max_len - 2:
                # 可能把答案截断
                item = item[:avg_len]
            if type == 'hotpot':
                sts_idx_local = 0
                for i in range(len(item)):
                    if item[i] == SEP_id:
                        sts2title[sts_idx] = idx2title[sp_list[idx]]
                        sts2idx[sts_idx] = sts_idx_local
                        sts_idx += 1
                        sts_idx_local += 1
            context_ids.extend(item)
        context_ids = context_ids[:max_len - 1] + [tokenizer.sep_token_id]
        pred_answer = None
        input_ids = torch.tensor(context_ids, dtype=torch.long, device=device).unsqueeze(0)
        attention_mask = torch.ones([1, len(context_ids)], dtype=torch.long, device=device)
        if type == 'hotpot':
            SEP_index = []
            for i in range(len(context_ids)):
                if context_ids[i] == SEP_id:
                    SEP_index.append(i)
            SEP_index = torch.LongTensor([SEP_index]).to(device)
            with torch.no_grad():
                outputs = qa_model(input_ids, attention_mask, sentence_index=SEP_index[0])
            sentence_select = torch.argmax(outputs['sentence_select'], dim=-1)
            assert sentence_select.shape[-1] == len(sts2idx)
            output_answer_type = outputs['output_answer_type']
            ans_type = torch.argmax(output_answer_type).item()
            if ans_type == 0:
                pred_answer = 'no'
            elif ans_type == 1:
                pred_answer = 'yes'
            sp = []
            sts_idx = 0
            for s in range(len(sentence_select)):
                if sentence_select[s] == 1:
                    sp.append([sts2title[s], sts2idx[s]])
            sp_pred[id] = sp
        else:
            with torch.no_grad():
                outputs = qa_model(input_ids, attention_mask)
        start_logits = outputs['start_qa_logits'][0]
        end_logits = outputs['end_qa_logits'][0]
        input_ids = input_ids[0]

        if pred_answer is None:
            if answer_merge:
                punc_token_list = tokenizer.convert_tokens_to_ids(['[CLS]', '?'])
                if type == 'hotpot':
                    punc_token_list.extend([SEP_id, DOC_id])
                span_id = merge_find_ans(start_logits, end_logits, input_ids.tolist(),punc_token_list, topk=topk)
                pred_answer = tokenizer.decode(span_id)
            else:
                all_tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())

                answer_tokens = all_tokens[torch.argmax(start_logits) : torch.argmax(end_logits) + 1]
                pred_answer = tokenizer.decode(
                    tokenizer.convert_tokens_to_ids(answer_tokens)
                )

        pred_answer = normalize_answer(pred_answer)
        if type == 'hotpot':
            ans_pred[id] = pred_answer

        else:
            pred_list.append({'id':id, 'predicted_answer': pred_answer, 'predicted_support_idxs': sp_list, 'predicted_answerable':True})
        
        if is_dev:
            ground_truth_answer = sample['answer']
            ground_truth_answer = normalize_answer(ground_truth_answer)

            em = compute_exact(ground_truth_answer, pred_answer)
            f1 = compute_f1(ground_truth_answer, pred_answer)
            em_tot.append(em)
            f1_tot.append(f1)

            if type == 'hotpot':
                sp_em, sp_f1 = update_sp(sp, sample['supporting_facts'])
                sp_em_tot.append(sp_em)
                sp_f1_tot.append(sp_f1)
                print(f"sp em:{sum(sp_em_tot) / len(sp_em_tot)}, sp f1:{sum(sp_f1_tot) / len(sp_f1_tot)}")
    if is_dev:
        print(f"em:{sum(em_tot) / len(em_tot)}, f1:{sum(f1_tot) / len(f1_tot)}")
        if type == 'hotpot':
            print(f"sp em:{sum(sp_em_tot) / len(sp_em_tot)}, sp f1:{sum(sp_f1_tot) / len(sp_f1_tot)}")
    if type == 'musique':
        with jsonlines.open(pred_filename, "w") as wfd:
            for data in pred_list:
                wfd.write(data)
    else:
        with open(pred_filename, "w", encoding="utf-8") as f:
            json.dump({"answer": ans_pred, "sp": sp_pred}, f, ensure_ascii=False, indent=4)
    print(f"evaluation finished!")
    torch.cuda.empty_cache()

if __name__ == '__main__':
    is_dev = True
    type = 'musique'
    # with open('project/hotpotqa/source_code/output/07-05-2023/train_2wiki_0-seed42-bsz8-fp16True-lr1e-05-decay0.0-warm0.1-valbsz1/pred_best.json', 'r') as f:
    #     retr_json = json.load(f)
    test_file_path = f"data/datasets/mrc/musique/musique_ans_v1.0_{'dev' if is_dev else 'test'}.jsonl"
    # test_file_path = f"data/datasets/mrc/2wikimultihop/data/{'dev' if is_dev else 'test'}.json"
    # test_raw_data = json.load(open(test_file_path))
    musique_data = open(test_file_path).readlines()
    test_raw_data = [json.loads(item) for item in musique_data]
    retr_json = get_retr_output(test_raw_data, is_dev=is_dev, type=type, beam_size=2)
    get_reader_qa_output(retr_json, test_raw_data, is_dev=is_dev, type=type, answer_merge=True, topk=10)