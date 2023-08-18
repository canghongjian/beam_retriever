from transformers import AutoModel, DebertaV2PreTrainedModel, DebertaV2Model, AutoModelForQuestionAnswering
import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss


class Reader(nn.Module):

    def __init__(self, config, model_name, task_type='hotpot', len_tokenizer=0, gradient_checkpointing=False):
        super().__init__()

        self.num_labels = config.num_labels if config.num_labels else 2
        self.config = config
        self.encoder = AutoModel.from_pretrained(model_name)
        self.qa_outputs = nn.Linear(config.hidden_size, self.num_labels)
        if task_type == 'hotpot':
            self.sentence_outputs = nn.Linear(config.hidden_size, 2)
            self.answer_typeout = nn.Linear(config.hidden_size, 3)
        if len_tokenizer > 0:
            # align the situation where tokenizer size gets bigger after adding special tokens
            self.encoder.resize_token_embeddings(len_tokenizer)
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

    def forward(
                self, 
                input_ids, 
                attention_mask, 
                start_pos=None, 
                end_pos=None,
                sentence_index=None,
                sentence_labels=None,
                answer_type=None
                ):
        '''

        :param input_ids:
        :param mask:
        :param token_type_ids:
        :return: three losses
        '''
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
        )
        sequence_output = outputs[0]
        B, L, E = sequence_output.size()
        
        device = sequence_output.device
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        ans_span_loss = None
        sentence_loss = None
        ans_type_loss = None

        qa_logits = self.qa_outputs(sequence_output)
        start_qa_logits, end_qa_logits = qa_logits.split(1, dim=-1)  # [1, 168, 1] separately
        start_qa_logits = start_qa_logits.squeeze(-1).contiguous()
        end_qa_logits = end_qa_logits.squeeze(-1).contiguous()

        ret = {
            'start_qa_logits': start_qa_logits,
            'end_qa_logits': end_qa_logits
        }
        
        if start_pos is not None and end_pos is not None:
            _, ignored_idx = start_qa_logits.shape
            loss_fct = CrossEntropyLoss(ignore_index=ignored_idx)
            start_pos = start_pos.clamp(0, ignored_idx)
            end_pos = end_pos.clamp(0, ignored_idx)
            start_loss = loss_fct(start_qa_logits, start_pos)  # [1, 168]  [1]
            end_loss = loss_fct(end_qa_logits, end_pos)
            ans_span_loss = (start_loss + end_loss) / 2
            ret.update({
                    'ans_span_loss': ans_span_loss
                })
        
        if sentence_index is not None:
            sentence_output = torch.index_select(sequence_output.reshape(B * L, E), 0, sentence_index)
            sentence_select = self.sentence_outputs(sentence_output)
            ret.update({
                'sentence_select': sentence_select
            })
            if sentence_labels is not None:
                loss_func = CrossEntropyLoss()
                sentence_loss = loss_func(sentence_select, sentence_labels)
                ret.update({
                    'sentence_loss': sentence_loss
                })

        if answer_type is not None:
            output_answer_type = self.answer_typeout(sequence_output[:, 0, :])
            ret.update({
                'output_answer_type': output_answer_type
            })
            loss_func = CrossEntropyLoss()
            ans_type_loss = loss_func(output_answer_type, answer_type)
            ret.update({
                'ans_type_loss': ans_type_loss
            })
        
        if sentence_loss is not None:
            loss = 0.2 * ans_span_loss + sentence_loss + ans_type_loss
        else:
            loss = ans_span_loss
        ret.update({
            'loss': loss
        })
        
        return ret
