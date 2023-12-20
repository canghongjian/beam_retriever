import os
import json

import requests, time
import re
import string
import collections
from tqdm import tqdm




api_key = ''
api_url = 'http://localhost:8000/v1/completions'
api_model_name = 'longchat-13b-16k' # 'gpt-3.5-turbo-16k'
headers={
    # 'Authorization': f'Bearer {api_key}',
    'Authorization': f'{api_key}',
    'Content-Type': 'application/json'
}

def get_response(tgt):
    data={
        "model": api_model_name,
        # "messages":[
        #     {
        #         "role":"user",
        #         "content":tgt
        #     }
        # ],
        "prompt": tgt,
        "max_tokens": 50,
        "temperature": 0.5,
    }
    response = requests.post(api_url, data=json.dumps(data), headers=headers)
    if response.status_code == 200:
        result = response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        result = 'error'
    return result


dataset_type = 'hotpot' 
llm_type = 'open_source' # open_source or openai
examples_num = 3 # few-shots
output_file_path = f"example_{dataset_type}_{llm_type}.jsonl"
output_file = open(output_file_path, 'w')

with open(f'prompts/few-shots_{dataset_type}.txt') as f:
    prompt_template_few = f.read().rstrip("\n")
with open(f'prompts/few-shots_{dataset_type}_br.txt') as f:
    prompt_template_few_br = f.read().rstrip("\n")     

if dataset_type == 'hotpot':
    pred_data = json.load(open('hotpot_pred_best_retr.json'))
    dev_data = json.load(open('data/hotpotqa/hotpot_dev_distractor_v1.json'))
    test_subsampled_data = open('data/processed_data/hotpotqa/test_subsampled.jsonl').readlines()
    test_subsampled_data = [json.loads(item) for item in test_subsampled_data]
    test_subsampled_data_ids = [item['question_id'] for item in test_subsampled_data]  
    test_subsampled_data = [item for item in dev_data if item['_id'] in test_subsampled_data_ids]

    for item in tqdm(test_subsampled_data):
        formatted_documents, formatted_documents_br = [], []
        question = item['question']
        documents, documents_br = [], []
        pred = pred_data[item['_id']]
        for idx, (title, sents) in enumerate(item['context']):
            document = {'title': title, 'text': "".join(sents)}
            documents.append(document)
            if idx in pred:
                documents_br.append(document)
            
        for document_index, document in enumerate(documents):
            formatted_documents.append(f"Document [{document_index+1}](Title: {document['title']}) {document['text']}")
        
        for document_index, document in enumerate(documents_br):
            formatted_documents_br.append(f"Document [{document_index+1}](Title: {document['title']}) {document['text']}")
        content = prompt_template_few.format(search_results="\n".join(formatted_documents), question=question)
        content_br = prompt_template_few_br.format(search_results="\n".join(formatted_documents_br), question=question)
        
        input = content
        try:
            response = get_response(input)
            if response != 'error':
                print("prompt tokens num:", response['usage']['prompt_tokens'])
                response = response['choices'][0]['message']['content'] if 'message' in response['choices'][0] else response['choices'][0]['text']
            else:
                print('error wait 20 seconds...')
                time.sleep(20)
                response = get_response(input)
                response = response['choices'][0]['message']['content'] if 'message' in response['choices'][0] else response['choices'][0]['text']
        except Exception as e:
            print(e)
            print('wait 20 seconds...')
            time.sleep(20)
            response = get_response(input)
            if response != 'error':
                print("prompt tokens num:", response['usage']['prompt_tokens'])
                response = response['choices'][0]['message']['content'] if 'message' in response['choices'][0] else response['choices'][0]['text']
        print(item['question'])
        print(item['answer'])
        print(response)
        
        input = content_br
        try:
            response2 = get_response(input)
            if response2 != 'error':
                print("prompt tokens num:", response2['usage']['prompt_tokens'])
                response2 = response2['choices'][0]['message']['content'] if 'message' in response2['choices'][0] else response2['choices'][0]['text']
            else:
                print('error wait 20 seconds...')
                time.sleep(20)
                response2 = get_response(input)
                response2 = response2['choices'][0]['message']['content'] if 'message' in response2['choices'][0] else response2['choices'][0]['text']
        except Exception as e:
            print(e)
            print('wait 20 seconds...')
            time.sleep(20)
            response2 = get_response(input)
            if response2 != 'error':
                print("prompt tokens num:", response2['usage']['prompt_tokens'])
                response2 = response2['choices'][0]['message']['content'] if 'message' in response2['choices'][0] else response2['choices'][0]['text']
        print(response2)
        print()

        output_file.write(json.dumps({'id': item['_id'], 'response': response, 'response2': response2}))
        output_file.write('\n')
elif dataset_type == 'musique':
    retr_url = 'musique_pred_best.json'
    dev_data = open('data/musique_ans_v1.0_dev.jsonl').readlines()
    dev_data = [json.loads(item) for item in dev_data]
    
    pred_data = json.load(open(retr_url))
    test_subsampled_data = open('data/processed_data/musique/test_subsampled.jsonl').readlines()
    
    test_subsampled_data = [json.loads(item) for item in test_subsampled_data]
    # test_subsampled_data_ids = [item['question_id'] for item in test_subsampled_data]  
    # test_subsampled_data = [item for item in dev_data if item['id'] in test_subsampled_data_ids]
    
    for item in tqdm(test_subsampled_data):
        formatted_documents, formatted_documents_br = [], []
        question = item['question_text']
        documents, documents_br = [], []
        pred = pred_data[item['question_id']]
        for idx, para in enumerate(item['contexts']):
            document = {'title': para['title'], 'text': para['paragraph_text']}
            documents.append(document)
            if idx in pred:
                documents_br.append(document)
            
        for document_index, document in enumerate(documents):
            formatted_documents.append(f"Document [{document_index+1}](Title: {document['title']}) {document['text']}")
        
        for document_index, document in enumerate(documents_br):
            formatted_documents_br.append(f"Document [{document_index+1}](Title: {document['title']}) {document['text']}")
        content = prompt_template_few.format(search_results="\n".join(formatted_documents), question=question)
        content_br = prompt_template_few_br.format(search_results="\n".join(formatted_documents_br), question=question)
        
        input = content
        try:
            response = get_response(input)
            if response != 'error':
                print("prompt tokens num:", response['usage']['prompt_tokens'])
                response = response['choices'][0]['message']['content'] if 'message' in response['choices'][0] else response['choices'][0]['text']
            else:
                print('error wait 20 seconds...')
                time.sleep(20)
                response = get_response(input)
                response = response['choices'][0]['message']['content'] if 'message' in response['choices'][0] else response['choices'][0]['text']
        except Exception as e:
            print(e)
            print('wait 20 seconds...')
            time.sleep(20)
            response = get_response(input)
            if response != 'error':
                print("prompt tokens num:", response['usage']['prompt_tokens'])
                response = response['choices'][0]['message']['content'] if 'message' in response['choices'][0] else response['choices'][0]['text']
        print(item['question_id'])
        print(item['answers_objects'][0]['spans'][0])
        print(response)
        
        input = content_br
        try:
            response2 = get_response(input)
            if response2 != 'error':
                print("prompt tokens num:", response2['usage']['prompt_tokens'])
                response2 = response2['choices'][0]['message']['content'] if 'message' in response2['choices'][0] else response2['choices'][0]['text']
            else:
                print('error wait 20 seconds...')
                time.sleep(20)
                response2 = get_response(input)
                response2 = response2['choices'][0]['message']['content'] if 'message' in response2['choices'][0] else response2['choices'][0]['text']
        except Exception as e:
            print(e)
            print('wait 20 seconds...')
            time.sleep(20)
            response2 = get_response(input)
            if response2 != 'error':
                print("prompt tokens num:", response2['usage']['prompt_tokens'])
                response2 = response2['choices'][0]['message']['content'] if 'message' in response2['choices'][0] else response2['choices'][0]['text']
        print(response2)
        print()

        output_file.write(json.dumps({'id': item['question_id'], 'response': response, 'response2': response2}))
        output_file.write('\n')
elif dataset_type == '2wiki':
    pred_data = json.load(open('2wiki_pred_best.json'))
    dev_data = json.load(open('data/2wiki/data_ids/dev.json'))
    test_subsampled_data = open('data/processed_data/2wikimultihopqa/test_subsampled.jsonl').readlines()
    test_subsampled_data = [json.loads(item) for item in test_subsampled_data]
    test_subsampled_data_ids = [item['question_id'] for item in test_subsampled_data]  
    test_subsampled_data = [item for item in dev_data if item['_id'] in test_subsampled_data_ids]

    for item in tqdm(test_subsampled_data):
        formatted_documents, formatted_documents_br = [], []
        question = item['question']
        documents, documents_br = [], []
        pred = pred_data[item['_id']]
        for idx, (title, sents) in enumerate(item['context']):
            document = {'title': title, 'text': "".join(sents)}
            documents.append(document)
            if idx in pred:
                documents_br.append(document)
            
        for document_index, document in enumerate(documents):
            formatted_documents.append(f"Document [{document_index+1}](Title: {document['title']}) {document['text']}")
        
        for document_index, document in enumerate(documents_br):
            formatted_documents_br.append(f"Document [{document_index+1}](Title: {document['title']}) {document['text']}")
        content = prompt_template_few.format(search_results="\n".join(formatted_documents), question=question)
        content_br = prompt_template_few_br.format(search_results="\n".join(formatted_documents_br), question=question)
        
        input = content
        try:
            response = get_response(input)
            if response != 'error':
                print("prompt tokens num:", response['usage']['prompt_tokens'])
                response = response['choices'][0]['message']['content'] if 'message' in response['choices'][0] else response['choices'][0]['text']
            else:
                print('error wait 20 seconds...')
                time.sleep(20)
                response = get_response(input)
                response = response['choices'][0]['message']['content'] if 'message' in response['choices'][0] else response['choices'][0]['text']
        except Exception as e:
            print(e)
            print('wait 20 seconds...')
            time.sleep(20)
            response = get_response(input)
            if response != 'error':
                print("prompt tokens num:", response['usage']['prompt_tokens'])
                response = response['choices'][0]['message']['content'] if 'message' in response['choices'][0] else response['choices'][0]['text']
        print(item['question'])
        print(item['answer'])
        print(response)
        
        input = content_br
        try:
            response2 = get_response(input)
            if response2 != 'error':
                print("prompt tokens num:", response2['usage']['prompt_tokens'])
                response2 = response2['choices'][0]['message']['content'] if 'message' in response2['choices'][0] else response2['choices'][0]['text']
            else:
                print('error wait 20 seconds...')
                time.sleep(20)
                response2 = get_response(input)
                response2 = response2['choices'][0]['message']['content'] if 'message' in response2['choices'][0] else response2['choices'][0]['text']
        except Exception as e:
            print(e)
            print('wait 20 seconds...')
            time.sleep(20)
            response2 = get_response(input)
            if response2 != 'error':
                print("prompt tokens num:", response2['usage']['prompt_tokens'])
                response2 = response2['choices'][0]['message']['content'] if 'message' in response2['choices'][0] else response2['choices'][0]['text']
        print(response2)
        print()

        output_file.write(json.dumps({'id': item['_id'], 'response': response, 'response2': response2}))
        output_file.write('\n')

output_file.close()