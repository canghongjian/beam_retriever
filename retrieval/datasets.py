import json
import random

import torch
from torch.utils.data import Dataset

import numpy as np


class HotpotQADataset(Dataset):

    def __init__(self, tokenizer, data_path, max_len=512, type='hotpot'):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.type = type
        self.max_passages_num = 25
        print("beginning to read data from " + data_path)
        if self.type.startswith('hotpot'):
            with open(data_path, 'r') as f:
                self.data = json.load(f)
        else:
            musique_train_data = open(data_path).readlines()
            self.data = [json.loads(item) for item in musique_train_data]
        print(f"Total sample count {len(self.data)}")

    def __getitem__(self, index):
        sample = self.data[index]
        question = sample['question'] if self.type != 'iirc' else (sample['question_text'] + sample['pinned_contexts'][0]['paragraph_text'])
        if question.endswith("?"):
            question = question[:-1]
        q_codes = self.tokenizer.encode(question, add_special_tokens=False, return_tensors="pt", truncation=True, max_length=self.max_len).squeeze(0)
        sp_title_set = set()
        c_codes = []
        sf_idx = []
        if self.type == 'hotpot':
            id = sample['_id']
            for sup in sample['supporting_facts']:
                sp_title_set.add(sup[0])
            for idx, (title, sentences) in enumerate(sample['context']):
                if title in sp_title_set:
                    sf_idx.append(idx)
                l = title + "".join(sentences)
                encoding = self.tokenizer.encode(l, add_special_tokens=False, return_tensors="pt", truncation=True, max_length=self.max_len-q_codes.shape[-1]).squeeze(0)
                c_codes.append(encoding)
        elif self.type == 'musique':
            # musique
            id = sample['id']
            for i, para in enumerate(sample['paragraphs']):
                # if para['is_supporting']:
                #     sf_idx.append(i)
                l = para['title'] + '.' + para['paragraph_text']
                encoding = self.tokenizer.encode(l, add_special_tokens=False, return_tensors="pt", truncation=True, max_length=self.max_len-q_codes.shape[-1]).squeeze(0)
                c_codes.append(encoding)
            # label order
            for item_json in sample['question_decomposition']:
                sf_idx.append(item_json['paragraph_support_idx'])
        elif self.type == 'iirc':
            id = sample['question_id']
            for i, para in enumerate(sample['contexts']):
                if i > self.max_passages_num:
                    break
                l = para['title'] + '.' + para['paragraph_text']
                encoding = self.tokenizer.encode(l, add_special_tokens=False, return_tensors="pt", truncation=True, max_length=self.max_len-q_codes.shape[-1]).squeeze(0)
                c_codes.append(encoding)
                if para['is_supporting']:
                    sf_idx.append(para['idx'])
        elif self.type == 'hotpot_reranker':
            id = sample['_id']
            for i, para in enumerate(sample['paragraphs']):
                l = para['title'] + '.' + para['paragraph_text']
                encoding = self.tokenizer.encode(l, add_special_tokens=False, return_tensors="pt", truncation=True, max_length=self.max_len-q_codes.shape[-1]).squeeze(0)
                c_codes.append(encoding)
                if para['is_supporting']:
                    sf_idx.append(i)
        
        res = {
            'q_codes': q_codes,
            'c_codes': c_codes,
            'sf_idx': sf_idx,
            'id': id,
        }
        return res

    def __len__(self):
        return len(self.data)

def collate_fn(samples):
    if len(samples) == 0:
        return {}
    batch = {
        'q_codes': [s['q_codes'] for s in samples],
        'c_codes': [s['c_codes'] for s in samples],
        "sf_idx": [s['sf_idx'] for s in samples],
        "id": [s['id'] for s in samples],
    }
    return batch

class HopDataset(Dataset):
    def __init__(self, tokenizer, data, max_len=512, is_training=True):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = data
        self.is_training = is_training
        print(f"Total sample count {len(self.data)}")

    def __getitem__(self, index):
        sample = self.data[index]
        inputs = sample['input']
        context = sample['context']
        label = sample['label']
        question = inputs[0]
        pre_passages = inputs[1:] if len(inputs) > 1 else []
        if self.is_training and len(pre_passages) > 1:
            # for training
            random.shuffle(pre_passages)
        if inputs[0].endswith("?"):
            question = question[:-1]
            inputs[0] = question
        question_codes = self.tokenizer.encode(question, add_special_tokens=False, truncation=True, max_length=self.max_len)
        
        mean_passage_length = (self.max_len - len(question_codes)) // (len(pre_passages) + 1)
        try:
            inputs_codes = self.tokenizer.encode("".join(inputs), add_special_tokens=False,truncation=True, max_length=self.max_len)
            context_codes = self.tokenizer.encode(context, add_special_tokens=False, truncation=True, max_length=self.max_len)
            if len(inputs_codes) + len(context_codes) > self.max_len:
                context_codes = context_codes[:mean_passage_length]
                if len(inputs_codes) + len(context_codes) > self.max_len:
                    pre_passages_codes = [self.tokenizer.encode(item, add_special_tokens=False,truncation=True, max_length=self.max_len) for item in pre_passages]
                    idx = 0
                    inputs_codes = question_codes[:]
                    while sum([len(item) for item in pre_passages_codes]) > self.max_len - len(question_codes):
                        pre_passages_codes[idx] = pre_passages_codes[idx][:mean_passage_length]
                        inputs_codes.extend(pre_passages_codes[idx])
                        idx += 1
            assert len(inputs_codes) + len(context_codes) <= self.max_len
        except Exception as e:
            print(e)
            print(f"question:{question}, len(question_codes):{len(question_codes)}, mean_passage_length:{mean_passage_length}, len(pre_passages):{len(pre_passages)}")
            print(f"len_inputs_code:{len(inputs_codes)}, len_context_codes:{len(context_codes)}")
            print(f"pre_passages_len:{[len(item) for item in pre_passages_codes]}, self.max_len - len(question_codes):{self.max_len - len(question_codes)}")
            raise e

        res = {
            'input_ids': torch.tensor(inputs_codes+ context_codes, dtype=torch.long),
            'label': label,
            'hop': len(pre_passages) + 1
        }
        return res

    def __len__(self):
        return len(self.data)
    
def collate_fn_each_hop(samples):
    if len(samples) == 0:
        return {}
    max_q_sp_len = max([item['input_ids'].shape[-1] for item in samples])
    all_q_doc_input_ids = torch.zeros((len(samples), max_q_sp_len), dtype=torch.long)
    all_q_doc_attention_mask = torch.zeros((len(samples), max_q_sp_len), dtype=torch.long)
    labels = torch.zeros(len(samples), dtype=torch.long)

    for i, sample in enumerate(samples):
        len_input_ids = sample['input_ids'].shape[-1]
        all_q_doc_input_ids[i, :len_input_ids] = sample['input_ids'].view(-1)
        all_q_doc_attention_mask[i, :len_input_ids] = 1
        labels[i] = sample['label']
    batch = {
        'input_ids': all_q_doc_input_ids,
        'attention_mask': all_q_doc_attention_mask,
        "labels": labels,
        "hops": [s['hop'] for s in samples]
    }
    return batch