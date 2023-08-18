import json
from tqdm import tqdm
import time
import requests
import re
import string
import collections

def gpt_turbo_request(content, url=None):
    headers = {"Content-Type": "application/json", "api-key": "asdadasdsa"}
    system_prompt = {'role': "system", "content": "You are a qa test machine, you need to answer the [Question] from given the [Context], you only need to come out the correct answer without anyother words."}
    data = {'messages':[system_prompt, {"role": "user", "content": content}], "model":"gpt-3.5-turbo"}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response = json.loads(response.text)['response']
    return response

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

if __name__ == '__main__':
    dataset_type = '2wiki'  

    all_passages_list = []
    pred_passages_list = []
    response_all_list = []
    response_pred_list = []
    output_response_list = []
    em_list1 = []
    f1_list1 = []
    em_list2 = []
    f1_list2 = []
    if dataset_type == 'musique':
        base_url = 'datasets/mrc/'
        musique_url = base_url + 'musique/'
        musique_dev_data = open(musique_url + 'musique_ans_v1.0_dev.jsonl').readlines()
        musique_dev_data = [json.loads(item) for item in musique_dev_data]
        musique_pred_data = json.load(open('pred_retr793_musique.json'))

        for item in musique_dev_data:
            idx2codes = {}
            c_codes = []
            total_contexts = ""
            for i, para in enumerate(item['paragraphs']):
                l = para['title'] + ' ' + para['paragraph_text']
                idx2codes[i] = l
                total_contexts += l

            for pred_idx in musique_pred_data[item['id']]:
                c_codes.append(idx2codes[pred_idx])
            all_passages_list.append((item['question'], total_contexts,item['answer']))
            pred_passages_list.append((item['question'], "".join(c_codes),item['answer']))

        for i in tqdm(range(len(musique_dev_data))):
            q = all_passages_list[i][0]
            c1 = all_passages_list[i][1]
            c2 = pred_passages_list[i][1]
            answer = all_passages_list[i][2]
            content1 = """[Question] : " {} " [Context]: " {} " """.format(q, c1)
            content2 = """[Question] : " {} " [Context]: " {} " """.format(q, c2)
            response = gpt_turbo_request(content1)
            time.sleep(1)
            response2 = gpt_turbo_request(content2)
            response_pred_list.append(response2)
            time.sleep(0.5)
            em1 = compute_exact(answer, response)
            f11 = compute_f1(answer, response)
            em_list1.append(em1)
            f1_list1.append(f11)
            em2 = compute_exact(answer, response2)
            f12 = compute_f1(answer, response2)
            em_list2.append(em2)
            f1_list2.append(f12)
            output_response_list.append({'id': musique_dev_data[i]['id'], 'all_c_answer':response, 'pred_c_answer':response2})
    else:
        url = 'download/processed_data/'
        dev_data = None
        pred_data = None
        if dataset_type == 'hotpot':
            dev_data = json.load(open('datasets/mrc/hotpotqa/hotpot_dev_distractor_v1.json'))
            pred_data = json.load(open('pred_test_hotpot_v0_retr.json'))
            test_subsampled_data = open(url + 'hotpotqa/test_subsampled.jsonl').readlines()

        elif dataset_type == '2wiki':
            dev_data = json.load(open('datasets/mrc/2wikimultihop/data/dev.json'))
            pred_data = json.load(open('retr_2wiki_999.json'))
            test_subsampled_data = open(url + '2wikimultihopqa/test_subsampled.jsonl').readlines()
        else:
            raise ValueError(f"unsupported dataset type:{dataset_type}")

        test_subsampled_data = [json.loads(item) for item in test_subsampled_data]
        test_subsampled_data_ids = [item['question_id'] for item in test_subsampled_data]  
        test_subsampled_data = [item for item in dev_data if item['_id'] in test_subsampled_data_ids]
        
        for sample in tqdm(test_subsampled_data):
            question = sample['question']
            answer = sample['answer']
            pred = pred_data[sample['_id']]
            contexts = []
            for i, (title, sts) in enumerate(sample['context']):
                contexts.append(title + ''.join(sts))
            pred_contexts = ""
            all_contexts = "".join(contexts)
            for pred_idx in pred:
                pred_contexts += contexts[pred_idx]
            content1 = """[Question] : " {} " [Context]: " {} " """.format(question, all_contexts)
            content2 = """[Question] : " {} " [Context]: " {} " """.format(question, pred_contexts)
            response = gpt_turbo_request(content1)
            time.sleep(1)
            response2 = gpt_turbo_request(content2)
            response_pred_list.append(response2)
            time.sleep(1)
            em1 = compute_exact(answer, response)
            f11 = compute_f1(answer, response)
            em_list1.append(em1)
            f1_list1.append(f11)
            em2 = compute_exact(answer, response2)
            f12 = compute_f1(answer, response2)
            em_list2.append(em2)
            f1_list2.append(f12)
            output_response_list.append({'id': sample['_id'], 'all_c_answer':response, 'pred_c_answer':response2})
    
    print(f"em_all:{sum(em_list1) / len(em_list1)}, f1_all:{sum(f1_list1) / len(f1_list1)}")
    print(f"em_pred:{sum(em_list2) / len(em_list2)}, f1_pred:{sum(f1_list2) / len(f1_list2)}")
    json.dump(output_response_list, open(f'output_{dataset_type}_answers.json', 'w'), ensure_ascii=False, indent=4)