from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import logging

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, MultiMarginLoss
import torch.nn.functional as F

from transformers import BertPreTrainedModel,RobertaConfig, AlbertModel
from customized_layers import OCN_Att_layer, OCN_CoAtt_layer, OCN_SelfAtt_layer, OCN_Merge_layer, QANet_Att_layer, SequenceSummary,SplitBertEncoder, BertModel
from customized_layers import ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP, RobertaModel, KVMem_Att_layer

class ModelForMCRC(BertPreTrainedModel):
    def __init__(self, config, model_name):
        super(ModelForMCRC, self).__init__(config)
        self.num_labels = config.num_labels
        if 'roberta' in model_name:
            print ('Building RoBERTa model')
            self.core = RobertaModel(config)
        elif 'albert' in model_name:
            print ('Building AlBert model')
            self.core = AlbertModel(config)
        elif 'bert' in model_name:
            print ('Building BERT model')
            self.core = BertModel(config)
        else:
            print ('did not recognize the model')
            exit(0)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if 'roberta' in model_name:
            self.classifier = RobertaClassificationHead(config)
        else:
            self.classifier = nn.Linear(config.hidden_size, 1)
        self.model_name = model_name
        self.init_weights()

    def redistribute(self, split):
        self.core.encoder.redistribute(split)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None,
                concepts=None, concepts_mask=None, concepts_mask_full=None):
        seq_length = input_ids.size(2)
        outputs = self.core(
            input_ids=input_ids.view(-1, seq_length),
            attention_mask=attention_mask.view(-1, seq_length), 
            token_type_ids=token_type_ids.view(-1, seq_length),
            position_ids=position_ids, 
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )
        if 'roberta' in self.model_name:
            sequence_output = outputs[0]
            logits = self.classifier(sequence_output)
        else:
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
        logits = logits.view(-1, self.num_labels)
        outputs = (logits,) + outputs[2:] 

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 1)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class OptionCompareCell(nn.Module):

    def __init__(self, config):
        super(OptionCompareCell, self).__init__()
        self.option_att_layer = OCN_Att_layer(config)
        self.option_merge_layer = OCN_Merge_layer(config)
        self.CoAtt_layer = OCN_CoAtt_layer(config)
        self.SelfAtt_layer = OCN_SelfAtt_layer(config)

    def forward(self, encoded_o, encoded_q, option_mask, question_mask):
        o1o2 = self.option_att_layer(encoded_o[:, 1, :, :], option_mask[:, 1, :], encoded_o[:, 0, :, :], option_mask[:, 0, :])
        o1o3 = self.option_att_layer(encoded_o[:, 2, :, :], option_mask[:, 2, :], encoded_o[:, 0, :, :], option_mask[:, 0, :])
        o1o4 = self.option_att_layer(encoded_o[:, 3, :, :], option_mask[:, 3, :], encoded_o[:, 0, :, :], option_mask[:, 0, :])
        o1o5 = self.option_att_layer(encoded_o[:, 4, :, :], option_mask[:, 4, :], encoded_o[:, 0, :, :], option_mask[:, 0, :])
        merged_o1 = self.option_merge_layer(encoded_o[:, 0, :, :], o1o2, o1o3, o1o4, o1o5, encoded_q[:, 0, :, :], question_mask[:, 0, :])
        reread_o1 = self.CoAtt_layer(encoded_q[:, 0, :, :], question_mask[:, 0, :], merged_o1, option_mask[:, 0, :])
        final_o1 = self.SelfAtt_layer(reread_o1, option_mask[:, 0, :], reread_o1, option_mask[:, 0, :])

        o2o1 = self.option_att_layer(encoded_o[:, 0, :, :], option_mask[:, 0, :], encoded_o[:, 1, :, :], option_mask[:, 1, :])
        o2o3 = self.option_att_layer(encoded_o[:, 2, :, :], option_mask[:, 2, :], encoded_o[:, 1, :, :], option_mask[:, 1, :])
        o2o4 = self.option_att_layer(encoded_o[:, 3, :, :], option_mask[:, 3, :], encoded_o[:, 1, :, :], option_mask[:, 1, :])
        o2o5 = self.option_att_layer(encoded_o[:, 4, :, :], option_mask[:, 4, :], encoded_o[:, 1, :, :], option_mask[:, 1, :])
        merged_o2 = self.option_merge_layer(encoded_o[:, 1, :, :], o2o1, o2o3, o2o4, o2o5, encoded_q[:, 1, :, :], question_mask[:, 1, :])
        reread_o2 = self.CoAtt_layer(encoded_q[:, 1, :, :], question_mask[:, 1, :], merged_o2, option_mask[:, 1, :])
        final_o2 = self.SelfAtt_layer(reread_o2, option_mask[:, 1, :], reread_o2, option_mask[:, 1, :])

        o3o1 = self.option_att_layer(encoded_o[:, 0, :, :], option_mask[:, 0, :], encoded_o[:, 2, :, :], option_mask[:, 2, :])
        o3o2 = self.option_att_layer(encoded_o[:, 1, :, :], option_mask[:, 1, :], encoded_o[:, 2, :, :], option_mask[:, 2, :])
        o3o4 = self.option_att_layer(encoded_o[:, 3, :, :], option_mask[:, 3, :], encoded_o[:, 2, :, :], option_mask[:, 2, :])
        o3o5 = self.option_att_layer(encoded_o[:, 4, :, :], option_mask[:, 4, :], encoded_o[:, 2, :, :], option_mask[:, 2, :])
        merged_o3 = self.option_merge_layer(encoded_o[:, 2, :, :], o3o1, o3o2, o3o4, o3o5, encoded_q[:, 2, :, :], question_mask[:, 2, :])
        reread_o3 = self.CoAtt_layer(encoded_q[:, 2, :, :], question_mask[:, 2, :], merged_o3, option_mask[:, 2, :])
        final_o3 = self.SelfAtt_layer(reread_o3, option_mask[:, 2, :], reread_o3, option_mask[:, 2, :])

        o4o1 = self.option_att_layer(encoded_o[:, 0, :, :], option_mask[:, 0, :], encoded_o[:, 3, :, :], option_mask[:, 3, :])
        o4o2 = self.option_att_layer(encoded_o[:, 1, :, :], option_mask[:, 1, :], encoded_o[:, 3, :, :], option_mask[:, 3, :])
        o4o3 = self.option_att_layer(encoded_o[:, 2, :, :], option_mask[:, 2, :], encoded_o[:, 3, :, :], option_mask[:, 3, :])
        o4o5 = self.option_att_layer(encoded_o[:, 4, :, :], option_mask[:, 4, :], encoded_o[:, 3, :, :], option_mask[:, 3, :])
        merged_o4 = self.option_merge_layer(encoded_o[:, 3, :, :], o4o1, o4o2, o4o3, o4o5, encoded_q[:, 3, :, :], question_mask[:, 3, :])
        reread_o4 = self.CoAtt_layer(encoded_q[:, 3, :, :], question_mask[:, 3, :], merged_o4, option_mask[:, 3, :])
        final_o4 = self.SelfAtt_layer(reread_o4, option_mask[:, 3, :], reread_o4, option_mask[:, 3, :])

        o5o1 = self.option_att_layer(encoded_o[:, 0, :, :], option_mask[:, 0, :], encoded_o[:, 4, :, :], option_mask[:, 4, :])
        o5o2 = self.option_att_layer(encoded_o[:, 1, :, :], option_mask[:, 1, :], encoded_o[:, 4, :, :], option_mask[:, 4, :])
        o5o3 = self.option_att_layer(encoded_o[:, 2, :, :], option_mask[:, 2, :], encoded_o[:, 4, :, :], option_mask[:, 4, :])
        o5o4 = self.option_att_layer(encoded_o[:, 3, :, :], option_mask[:, 3, :], encoded_o[:, 4, :, :], option_mask[:, 4, :])
        merged_o5 = self.option_merge_layer(encoded_o[:, 4, :, :], o5o1, o5o2, o5o3, o5o4, encoded_q[:, 4, :, :], question_mask[:, 4, :])
        reread_o5 = self.CoAtt_layer(encoded_q[:, 4, :, :], question_mask[:, 4, :], merged_o5, option_mask[:, 4, :])
        final_o5 = self.SelfAtt_layer(reread_o5, option_mask[:, 4, :], reread_o5, option_mask[:, 4, :])

        candidates = torch.cat([final_o1.unsqueeze(1), final_o2.unsqueeze(1), final_o3.unsqueeze(1), final_o4.unsqueeze(1), final_o5.unsqueeze(1)], dim=1)
        candidates, _ = torch.max(candidates, dim=2)
        return candidates

class OCNModel(BertPreTrainedModel):

    def __init__(self, config, model_name):
        super(OCNModel, self).__init__(config)
        self.num_labels = config.num_labels

        if 'roberta' in model_name:
            print ('Building RoBERTa model')
            self.core = RobertaModel(config)
        elif 'albert' in model_name:
            print ('Building AlBert model')
            self.core = AlbertModel(config)
        elif 'bert' in model_name:
            print ('Building BERT model')
            self.core = BertModel(config)
        else:
            print ('did not recognize the model')
            exit(0)
        
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.loss_fct = CrossEntropyLoss()
        self.OCN_cell = OptionCompareCell(config)
        self.hidden_size = config.hidden_size
        self.inject = False
        if 'inj' in model_name:
            self.Inject_layer = KVMem_Att_layer(config)
            self.inject = True 
        self.init_weights()

    def redistribute(self, split):
        self.core.encoder.redistribute(split)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None, concepts=None, concepts_mask=None, concepts_mask_full=None):
        batch_size, num_cand, seq_length = input_ids.shape
        outputs = self.core(input_ids.view(-1, seq_length),
                               attention_mask=attention_mask.view(-1, seq_length),
                               token_type_ids=token_type_ids.view(-1,seq_length),
                               position_ids=position_ids,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds)

        all_hidden = outputs[0]
        if self.inject:
            path_len = concepts.size(3)
            cs_embeddings = self.core.embeddings(concepts.view(-1, path_len))
            concepts_mask = concepts_mask.view(batch_size*num_cand, -1)
            concepts_mask_full = concepts_mask_full.view(-1, path_len)
            all_hidden = self.Inject_layer(outputs[0], cs_embeddings, attention_mask.view(-1,seq_length), concepts_mask, concepts.shape)

        option_mask = (token_type_ids >= 1).long()
        question_mask = attention_mask - option_mask

        encoded = all_hidden.view(batch_size, num_cand, seq_length, self.hidden_size)
        encoded_o = encoded * option_mask.unsqueeze(3).float()
        encoded_q = encoded * question_mask.unsqueeze(3).float()
        candidates = self.OCN_cell(encoded_o, encoded_q, option_mask, question_mask)
        
        logits = self.classifier(candidates).squeeze(2)
        outputs = (logits,) + outputs[2:]
        
        if labels is not None:
            loss = self.loss_fct(logits, labels.view(-1))
            outputs = (loss,) + outputs

        return outputs
