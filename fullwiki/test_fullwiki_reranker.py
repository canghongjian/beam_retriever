from retriever_model import Retriever
from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch
def load_saved(model, path, exact=True):
    try:
        state_dict = torch.load(path)
    except:
        state_dict = torch.load(path, map_location=torch.device('cpu'))

    state_dict = {k: v for (k, v) in state_dict.items()}
    def filter(x):
        return x[7:] if x.startswith('module.') else x

    if exact:
        state_dict = {filter(k): v for (k, v) in state_dict.items()}
    else:
        state_dict = {filter(k): v for (k, v) in state_dict.items() if filter(k) in model.state_dict()}
    model.load_state_dict(state_dict)
    return model


ckpt_url = ''
model_path = 'models/deberta-v3-large'
device = torch.device("cuda", 4)
tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)
config.cls_token_id = tokenizer.cls_token_id
config.sep_token_id = tokenizer.sep_token_id
re_model = Retriever(config, model_path, encoder_class=AutoModel, mean_passage_len=120, beam_size=1, gradient_checkpointing=True)
re_model = load_saved(re_model, ckpt_url)
re_model = re_model.to(device)
re_model.eval()

from tqdm import tqdm
beam_retrieval_pred_em = []
# beam_retrieval_pred_f1 = []
recall_num = 100
max_len = 512
dev_data = mdr_train_qa_dev
# a = [item['title'] for item in br_train_dev_data[i]['paragraphs']]
#     sf = [item['title'] for item in mdr_train_qa_dev[i]['sp']]
#     b = []
#     for item in mdr_train_qa_dev[i]['candidate_chains']:
#         for item2 in item:
#             b.append(item2['title'])
#     hit1.append(int(len(set(b) & set(sf)) == 2))
#     hit2.append(int(len(set(b[:2]) & set(sf)) == 2))
for i, item in tqdm(enumerate(dev_data)):
    title2doc = {}
    sf = [item2['title'] for item2 in dev_data[i]['sp']]
    for item2 in dev_data[i]['candidate_chains']:
        for item3 in item2:
            title2doc[item3['title']] = item3['sents']
            if len(title2doc) >= recall_num:
                break
        if len(title2doc) >= recall_num:
            break
    candidate_set = []
    sf_idx = []
    sp_title_set = set(sf)
    idx = 0
    for k, v in title2doc.items():
        if k in sf:
            sf_idx.append(idx)
        idx += 1
        candidate_set.append({'title': k, 'sents': v})
    question = item['question']
    if question.endswith("?"):
        question = question[:-1]
    q_codes = tokenizer.encode(question, add_special_tokens=False, return_tensors="pt", truncation=True, max_length=max_len).squeeze(0)
    c_codes = []
    for item2 in candidate_set:
        l = item2['title'] + "".join(item2['sents'])
        encoding = tokenizer.encode(l, add_special_tokens=False, return_tensors="pt", truncation=True, max_length=max_len-q_codes.shape[-1]).squeeze(0)
        encoding = encoding.to(device)
        c_codes.append(encoding)
    q_codes = q_codes.to(device)
    q_codes_input = [q_codes]
    c_codes_input = [c_codes]
    hop = 2
    with torch.no_grad():
        current_preds = re_model(q_codes_input, c_codes_input, hop=hop, sf_idx=[])
    pred_titles = [candidate_set[item2]['title'] for item2 in current_preds['current_preds'][0]]
#     print(pred_titles, sp_title_set)
    if len(set(pred_titles) & sp_title_set) == 2:
        beam_retrieval_pred_em.append(1)
    else:
        beam_retrieval_pred_em.append(0)
    if i % 50 == 0:
        print(sum(beam_retrieval_pred_em) / len(beam_retrieval_pred_em))