import json
import random
import unicodedata

import torch
from torch.utils.data import Dataset

import numpy as np

def get_next(s):
    j = -1
    next_arr = [0] * len(s)
    next_arr[0] = j
    for i in range(1, len(s)):
        while j >= 0 and s[i] != s[j+1]:
            j = next_arr[j]
        if s[i] == s[j+1]:
            j += 1
        next_arr[i] = j
    return next_arr

def kmp(src, tgt):
    if len(tgt) == 0 or len(tgt) > len(src):
        return -1
    j = -1
    next_arr = get_next(tgt)
    for i in range(len(src)):
        while j >= 0 and src[i] != tgt[j+1]:
            j = next_arr[j]
        if src[i] == tgt[j+1]:
            j += 1
        if j == len(tgt) - 1:
            return i - len(tgt) + 1
    return -1

class MHReaderDataset(Dataset):

    def __init__(self, 
            tokenizer, 
            data_path, 
            max_len=512, 
            type='hotpot',
            is_train=False):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.type = type
        self.is_train = is_train
        print("beginning to read data from " + data_path)
        if self.type == 'hotpot':
            with open(data_path, 'r') as f:
                self.data = json.load(f)
        elif self.type == 'musique':
            musique_train_data = open(data_path).readlines()
            self.data = [json.loads(item) for item in musique_train_data]
        print(f"Total sample count {len(self.data)}")

    def __getitem__(self, index):
        sample = self.data[index]
        question = sample['question']
        if question.endswith("?"):
            question = question[:-1]
        answer = sample['answer']
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)
        q_codes = self.tokenizer.encode(question, add_special_tokens=False, truncation=True, max_length=self.max_len)
        sp_title_set = {}
        c_codes = []
        if self.type == 'hotpot':
            # hotpot format
            DOC, SEP = "</d>", "</e>"
            SEP_id = self.tokenizer.convert_tokens_to_ids(SEP)
            DOC_id = self.tokenizer.convert_tokens_to_ids(DOC)
            for sup in sample['supporting_facts']:
                if sup[0] not in sp_title_set:
                    sp_title_set[sup[0]] = []
                sp_title_set[sup[0]].append(sup[1])
            for idx, (title, sentences) in enumerate(sample['context']):
                if title in sp_title_set:
                    context = title
                    for idx2, c in enumerate(sentences):
                        if idx2 in sp_title_set[title]:
                            context += (DOC + " " + c)
                        else:
                            context += (SEP + " " + c)
                    encoding = self.tokenizer.encode(context, add_special_tokens=False, truncation=True, max_length=self.max_len-len(q_codes))
                    c_codes.append(encoding)
        elif self.type == 'musique':
            # musique
            idx2codes = {}
            for i, para in enumerate(sample['paragraphs']):
                l = para['title'] + '.' + para['paragraph_text']
                encoding = self.tokenizer.encode(l, add_special_tokens=False, truncation=True, max_length=self.max_len-len(q_codes))
                idx2codes[i] = encoding

            # label order
            for item_json in sample['question_decomposition']:
                c_codes.append(idx2codes[item_json['paragraph_support_idx']])
        if self.is_train:
            # shuffle the related documents to improve the robustness of model
            random.shuffle(c_codes)
        total_len = len(q_codes) + sum([len(item) for item in c_codes])
        context_ids = [self.tokenizer.cls_token_id] + q_codes
        avg_len = (self.max_len - 2 - len(q_codes)) // len(c_codes)
        sentence_label = []
        for item in c_codes:
            if total_len > self.max_len - 2:
                # may truncate the answer
                start_idx = kmp(item, answer_ids)
                # if model max_position_embeddings was longer, the performance would be higher
                if not self.is_train or (start_idx == -1 or start_idx + len(answer_ids) <= avg_len):
                    item = item[:avg_len]
                else:
                    item = item[start_idx + len(answer_ids) - avg_len: start_idx + len(answer_ids)]
            if self.type == 'hotpot':
                for i in range(len(item)):
                    if item[i] == DOC_id:
                        sentence_label.append(1)
                        item[i] = SEP_id
                    elif item[i] == SEP_id:
                        sentence_label.append(0)
            context_ids.extend(item)
        context_ids = context_ids[:self.max_len - 1] + [self.tokenizer.sep_token_id]
        
        start_pos = kmp(context_ids, answer_ids)
        end_pos = start_pos + len(answer_ids) - 1

        res = {
            "input_ids": context_ids,
            "start_pos": start_pos,
            "end_pos": end_pos,
            "id": sample['id'] if self.type == 'musique' else sample['_id'],
            "answer": answer

        }
        if self.type == 'hotpot':
            # notice that input_ids are truncated by max_seq_len, possibly truncate the last few </e> in raw data
            SEP_index = []
            for i in range(len(context_ids)):
                if context_ids[i] == SEP_id:
                    SEP_index.append(i)
            assert len(sentence_label) == len(SEP_index)
            answer_type = 2
            if answer == "no":
                answer_type = 0
            elif answer == "yes":
                answer_type = 1
            
            res.update({
            "sentence_index": SEP_index,
            "sentence_labels": sentence_label,
            "answer_type": answer_type,
            "sentence_num": len(SEP_index)
            })
        return res

    def __len__(self):
        return len(self.data)


def reader_mhop_collate(samples):
    if len(samples) == 0:
        return {}
    # 组建tensors
    max_q_pp_len = max([len(s['input_ids']) for s in samples])
    batch_input_ids = torch.zeros((len(samples), max_q_pp_len),  dtype=torch.long)
    batch_attention_mask = torch.zeros((len(samples), max_q_pp_len), dtype=torch.long)
    batch_start_pos = torch.tensor([s['start_pos'] for s in samples], dtype=torch.long).reshape(len(samples))
    batch_end_pos = torch.tensor([s['end_pos'] for s in samples], dtype=torch.long).reshape(len(samples))

    for i, s in enumerate(samples):
        len_input_ids = len(s['input_ids'])
        batch_input_ids[i, :len_input_ids] = torch.tensor(s['input_ids'])
        batch_attention_mask[i, :len_input_ids] = 1

    batch = {
        'input_ids': batch_input_ids,
        'attention_mask': batch_attention_mask,
        'start_pos': batch_start_pos,
        'end_pos': batch_end_pos,
        "id": [s['id'] for s in samples],
        "answer": [s['answer'] for s in samples]
    }
    if 'sentence_labels' in samples[0]:
        # hotpot format
        batch_sentence_labels = torch.tensor(list(np.concatenate([s['sentence_labels'] for s in samples])),
                                             dtype=torch.long)
        batch_answer_type = torch.tensor([s['answer_type'] for s in samples], dtype=torch.long).reshape(len(samples))
        sentence_index = [s['sentence_index'] for s in samples]
        for i in range(1, len(sentence_index)):
            for j in range(len(sentence_index[i])):
                sentence_index[i][j] += max_q_pp_len * i
        batch.update({
        'sentence_index': torch.tensor(list(np.concatenate(sentence_index)), dtype=torch.long),
        'sentence_labels': batch_sentence_labels,
        'answer_type': batch_answer_type,
        'sentence_num': [s['sentence_num'] for s in samples]
        })

    return batch
