import torch.nn as nn
import torch
import math
import random


import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # 计算权重（alpha）
        p_t = torch.exp(-ce_loss)
        alpha_t = self.alpha * (1 - p_t)

        # 计算Focal Loss
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * ce_loss

        # 根据reduction参数选择损失的计算方式
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        elif self.reduction == 'none':
            return focal_loss
        else:
            raise ValueError("Unsupported reduction mode. Use 'mean', 'sum', or 'none'.")

class Retriever(nn.Module):

    def __init__(self,
                 config,
                 model_name,
                 encoder_class,
                 max_seq_len=512,
                 mean_passage_len=70,
                 beam_size=1,
                 gradient_checkpointing=False,
                 use_label_order=False,
                 use_negative_sampling=False,
                 use_focal=False,
                 use_early_stop=True,
                 ):
        super().__init__()
        self.encoder = encoder_class.from_pretrained(model_name, config=config)
        self.config = config
        self.max_seq_len = max_seq_len
        self.mean_passage_len = mean_passage_len # deprecated
        self.beam_size = beam_size
        self.gradient_checkpointing = gradient_checkpointing
        self.use_label_order = use_label_order # the label order is given for musique
        self.use_negative_sampling = use_negative_sampling if beam_size > 1 else False # whether use negative sampling, deprecated
        self.use_focal = use_focal
        self.use_early_stop = use_early_stop
        self.hop_classifier_layer = nn.Linear(config.hidden_size, 2)
        self.hop_n_classifier_layer = nn.Linear(config.hidden_size, 2)
        # self.hop1_classifier_layer = nn.Linear(config.hidden_size, 2)
        # self.hop2_classifier_layer = nn.Linear(config.hidden_size, 2)
        # self.hop3_classifier_layer = nn.Linear(config.hidden_size, 2)
        # self.hop4_classifier_layer = nn.Linear(config.hidden_size, 2)
        if self.gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

    def get_negative_sampling_results(self, context_ids, current_preds, sf_idx):
        closest_power_of_2 = 2 ** math.floor(math.log2(self.beam_size))
        powers = torch.arange(1, 1 + closest_power_of_2,  dtype=torch.int32)
        slopes = torch.pow(0.5, powers)
        each_sampling_nums = [max(1, int(len(context_ids) * item)) for item in slopes]
        last_pred_idx = set()
        sampled_set = {}
        for i in range(self.beam_size):
            last_pred_idx.add(current_preds[i][-1])
            sampled_set[i] = []
            for j in range(len(context_ids)):
                if j in current_preds[i] or j in last_pred_idx:
                    continue
                if set(current_preds[i] + [j]) == set(sf_idx):
                    continue
                sampled_set[i].append(j)
            random.shuffle(sampled_set[i])
            sampled_set[i] = sampled_set[i][:each_sampling_nums[i]]
        return sampled_set

    def forward(self, q_codes, c_codes, sf_idx, hop=0):
        '''
        hop predefined
        '''
        device = q_codes[0].device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        # the input ids of predictions and questions remained by last hop
        last_prediction = None
        pre_question_ids = None
        loss_function = nn.CrossEntropyLoss()
        focal_loss_function = None
        if self.use_focal:
            focal_loss_function = FocalLoss()
        question_ids = q_codes[0]
        context_ids = c_codes[0]
        current_preds = []
        if self.training:
            sf_idx = sf_idx[0]
            sf = sf_idx
            hops = len(sf)
        else:
            hops = hop if hop > 0 else len(sf_idx[0])
        if len(context_ids) <= hops or hops < 1:
            return {'current_preds': [list(range(hops))],
               'loss': total_loss}
        mean_passage_len = (self.max_seq_len - 2 - question_ids.shape[-1]) // hops
        for idx in range(hops):
            if idx == 0:
                # first hop
                qp_len = [min(self.max_seq_len - 2 - (hops - 1 - idx) * mean_passage_len, question_ids.shape[-1]+c.shape[-1]) for c in context_ids]
                next_question_ids = []
                hop1_qp_ids = torch.zeros([len(context_ids), max(qp_len) + 2], device=device, dtype=torch.long)
                hop1_qp_attention_mask = torch.zeros([len(context_ids), max(qp_len) + 2], device=device, dtype=torch.long)
                if self.training:
                    hop1_label = torch.zeros([len(context_ids)], dtype=torch.long, device=device)
                for i in range(len(context_ids)):
                    this_question_ids = torch.cat((question_ids, context_ids[i]))[:qp_len[i]]
                    hop1_qp_ids[i, 1:qp_len[i]+1] = this_question_ids.view(-1)
                    hop1_qp_ids[i, 0] = self.config.cls_token_id
                    hop1_qp_ids[i, qp_len[i]+1] = self.config.sep_token_id
                    hop1_qp_attention_mask[i, :qp_len[i]+1] = 1
                    if self.training:
                        if self.use_label_order:
                            if i == sf_idx[0]:
                                hop1_label[i] = 1
                        else:
                            if i in sf_idx:
                                hop1_label[i] = 1 
                    next_question_ids.append(this_question_ids)
                hop1_encoder_outputs = self.encoder(input_ids=hop1_qp_ids, attention_mask=hop1_qp_attention_mask)[0][:, 0, :] # [doc_num, hidden_size]
                if self.training and self.gradient_checkpointing:
                    hop1_projection = torch.utils.checkpoint.checkpoint(self.hop_classifier_layer, hop1_encoder_outputs) # [doc_num, 2]
                else:
                    hop1_projection = self.hop_classifier_layer(hop1_encoder_outputs) # [doc_num, 2]
                
                if self.training:
                    total_loss = total_loss + loss_function(hop1_projection, hop1_label)
                _, hop1_pred_documents = hop1_projection[:, 1].topk(self.beam_size, dim=-1)
                last_prediction = hop1_pred_documents # used for taking new_question_ids
                pre_question_ids = next_question_ids
                current_preds = [[item.item()] for item in hop1_pred_documents] # used for taking the orginal passage index of the current passage
            else:
                # set up the vectors outside the beam_size loop
                qp_len_total = {}
                max_qp_len = 0
                last_pred_idx = set()
                if self.training:
                    # stop predicting if the current hop's predictions are wrong
                    flag = False
                    for i in range(self.beam_size):
                        if self.use_label_order:
                            if current_preds[i][-1] == sf_idx[idx - 1]:
                                flag = True
                                break
                        else:
                            if set(current_preds[i]) == set(sf_idx[:idx]):
                                flag = True
                                break
                    if not flag and self.use_early_stop:
                        break
                for i in range(self.beam_size):
                    # expand the search space, and self.beam_size is the number of predicted passages
                    pred_doc = last_prediction[i]
                    # avoid iterativing over a duplicated passage, for example, it should be 9+8 instead of 9+9
                    last_pred_idx.add(current_preds[i][-1])
                    new_question_ids = pre_question_ids[pred_doc]
                    qp_len = {}
                    # obtain the sequence length which can be formed into the vector
                    for j in range(len(context_ids)):
                        if j in current_preds[i] or j in last_pred_idx:
                            continue 
                        qp_len[j] = min(self.max_seq_len - 2 - (hops - 1 - idx) * mean_passage_len, new_question_ids.shape[-1]+context_ids[j].shape[-1])
                        max_qp_len = max(max_qp_len, qp_len[j])
                    qp_len_total[i] = qp_len
                if len(qp_len_total) < 1:
                    # skip if all the predictions in the last hop are wrong 
                    break
                if self.use_negative_sampling and self.training:
                    # deprecated
                    current_sf = [sf_idx[idx]] if self.use_label_order else sf_idx
                    sampled_set = self.get_negative_sampling_results(context_ids, current_preds, sf_idx[:idx+1])
                    vector_num = 1 
                    for k in range(self.beam_size):
                        vector_num += len(sampled_set[k])
                else:
                    vector_num = sum([len(v) for k, v in qp_len_total.items()])
                # set up the vectors
                hop_qp_ids = torch.zeros([vector_num, max_qp_len + 2], device=device, dtype=torch.long)
                hop_qp_attention_mask = torch.zeros([vector_num, max_qp_len + 2], device=device, dtype=torch.long)
                if self.training:
                    hop_label = torch.zeros([vector_num], dtype=torch.long, device=device)
                vec_idx = 0
                pred_mapping = []
                next_question_ids = []
                last_pred_idx = set()

                for i in range(self.beam_size):
                    # expand the search space, and self.beam_size is the number of predicted passages
                    pred_doc = last_prediction[i]
                    # avoid iterativing over a duplicated passage, for example, it should be 9+8 instead of 9+9
                    last_pred_idx.add(current_preds[i][-1])
                    new_question_ids = pre_question_ids[pred_doc]
                    for j in range(len(context_ids)):
                        if j in current_preds[i] or j in last_pred_idx:
                            continue
                        if self.training and self.use_negative_sampling:
                            if j not in sampled_set[i] and not (set(current_preds[i] + [j]) == set(sf_idx[:idx+1])):
                                continue
                        # shuffle the order between documents
                        pre_context_ids = new_question_ids[question_ids.shape[-1]:].clone().detach()
                        context_list = [pre_context_ids, context_ids[j]]
                        if self.training:
                            random.shuffle(context_list)
                        this_question_ids = torch.cat((question_ids, torch.cat((context_list[0], context_list[1]))))[:qp_len_total[i][j]]
                        next_question_ids.append(this_question_ids)
                        hop_qp_ids[vec_idx, 1:qp_len_total[i][j]+1] = this_question_ids
                        hop_qp_ids[vec_idx, 0] = self.config.cls_token_id
                        hop_qp_ids[vec_idx, qp_len_total[i][j]+1] = self.config.sep_token_id
                        hop_qp_attention_mask[vec_idx, :qp_len_total[i][j]+1] = 1
                        if self.training:
                            if self.use_negative_sampling:
                                if set(current_preds[i] + [j]) == set(sf_idx[:idx+1]):
                                    hop_label[vec_idx] = 1
                            else:
                                # if self.use_label_order:
                                if set(current_preds[i] + [j]) == set(sf_idx[:idx+1]):
                                    hop_label[vec_idx] = 1
                                # else:
                                #     if j in sf_idx:
                                #         hop_label[vec_idx] = 1 
                        pred_mapping.append(current_preds[i] + [j])
                        vec_idx += 1

                assert len(pred_mapping) == hop_qp_ids.shape[0]
                hop_encoder_outputs = self.encoder(input_ids=hop_qp_ids, attention_mask=hop_qp_attention_mask)[0][:, 0, :] # [vec_num, hidden_size]
                # if idx == 1:
                #     hop_projection_func = self.hop2_classifier_layer
                # elif idx == 2:
                #     hop_projection_func = self.hop3_classifier_layer
                # else:
                #     hop_projection_func = self.hop4_classifier_layer
                hop_projection_func = self.hop_n_classifier_layer
                if self.training and self.gradient_checkpointing:
                    hop_projection = torch.utils.checkpoint.checkpoint(hop_projection_func, hop_encoder_outputs) # [vec_num, 2]
                else:
                    hop_projection = hop_projection_func(hop_encoder_outputs) # [vec_num, 2]
                if self.training:
                    if not self.use_focal:
                        total_loss = total_loss + loss_function(hop_projection, hop_label)
                    else:
                        total_loss = total_loss + focal_loss_function(hop_projection, hop_label)
                _, hop_pred_documents = hop_projection[:, 1].topk(self.beam_size, dim=-1)
                last_prediction = hop_pred_documents
                pre_question_ids = next_question_ids
                current_preds = [pred_mapping[hop_pred_documents[i].item()] for i in range(self.beam_size)]

        res = {'current_preds': current_preds,
               'loss': total_loss}
        return res



class SingleHopRetriever(nn.Module):

    def __init__(self, config, model_name, encoder_class):
        super().__init__()

        self.encoder = encoder_class.from_pretrained(model_name)
        self.project = nn.Linear(config.hidden_size, 2)

    def encode_seq(self, input_ids, mask):
        cls_rep = self.encoder(input_ids, mask)[0][:, 0, :]
        vector = self.project(cls_rep)
        return vector

    def forward(self, input_ids, mask):
        '''

        :param batch:
        ['q_input_ids]
        ['q_mask']
        ['all_q_doc_input_ids'] : [batch_size, doc_num, max_q_sp_len]
        :return:
        '''
        q = self.encode_seq(input_ids, mask)
        return q
    
