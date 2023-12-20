import random, json
from tqdm import tqdm

mdr_train_qa_dev = open('data/hotpotqa/mdr/dev_retrieval_b50_k50_sp.json').readlines()
mdr_train_qa_dev = [json.loads(item) for item in mdr_train_qa_dev]
mdr_train_qa_train = open('data/hotpotqa/mdr/train_retrieval_b100_k100_sp.json').readlines()
mdr_train_qa_train = [json.loads(item) for item in mdr_train_qa_train]

br_reranker_train_data = []
br_reranker_dev_data = []
passage_num = 48 # number of negative passages
for i in range(len(mdr_train_dev_data)):
    sf2doc = {item['title']: "".join(item['sents']) for item in mdr_train_qa_dev[i]['sp']}
    title2doc = {}
    for item in mdr_train_qa_dev[i]['candidate_chains']:
        for item2 in item:
            if item2['title'] not in title2doc:
                title2doc[item2['title']] = "".join(item2['sents'])
    for k, v in sf2doc.items():
        if k not in title2doc:
            title2doc[k] = v
    pos_paras, neg_paras = [], []
    for k, v in title2doc.items():
        if k in sf2doc:
            pos_paras.append({'title': k, 'paragraph_text': v, 'is_supporting': True})
        else:
            neg_paras.append({'title': k, 'paragraph_text': v, 'is_supporting': False})
    assert len(pos_paras) == 2
    random.shuffle(neg_paras)
    tmp = {}
    for k, v in br_train_dev_data[i].items():
        if k != 'paragraphs':
            tmp[k] = v
        else:
            to_append_paras = pos_paras + neg_paras[:passage_num]
            random.shuffle(to_append_paras)
            tmp[k] = to_append_paras
    br_reranker_dev_data.append(tmp)

for i in tqdm(range(len(mdr_train_qa_train))):
    sf2doc = {item['title']: "".join(item['sents']) for item in mdr_train_qa_train[i]['sp']}
    title2doc = {}
    for item in mdr_train_qa_train[i]['candidate_chains']:
        for item2 in item:
            if item2['title'] not in title2doc:
                title2doc[item2['title']] = "".join(item2['sents'])
    for k, v in sf2doc.items():
        if k not in title2doc:
            title2doc[k] = v
    pos_paras, neg_paras = [], []
    for k, v in title2doc.items():
        if k in sf2doc:
            pos_paras.append({'title': k, 'paragraph_text': v, 'is_supporting': True})
        else:
            neg_paras.append({'title': k, 'paragraph_text': v, 'is_supporting': False})
    assert len(pos_paras) == 2
    random.shuffle(neg_paras)
    tmp = {}
    for k, v in br_train_train_data[i].items():
        if k != 'paragraphs':
            tmp[k] = v
        else:
            to_append_paras = pos_paras + neg_paras[:passage_num]
            random.shuffle(to_append_paras)
            tmp[k] = to_append_paras
    br_reranker_train_data.append(tmp)
    
json.dump(br_reranker_train_data, open('data/hotpotqa/hotpotqa_fullwiki_br_train_v2_50.json', 'w'), ensure_ascii=False, indent=4)
json.dump(br_reranker_dev_data, open('data/hotpotqa/hotpotqa_fullwiki_br_dev_v2_50.json', 'w'), ensure_ascii=False, indent=4)
