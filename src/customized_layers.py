from __future__ import (absolute_import, division, print_function,
						unicode_literals)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from torch.nn import Identity
from transformers import BertPreTrainedModel,RobertaConfig

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
}

def gelu(x):
	"""Implementation of the gelu activation function.
		For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
		0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
		Also see https://arxiv.org/abs/1606.08415
	"""
	return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
	return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


# try:
# 	from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
# except (ImportError, AttributeError) as e:
# 	logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
# 	BertLayerNorm = torch.nn.LayerNorm

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class BertSelfAttention(nn.Module):
	def __init__(self, config):
		super(BertSelfAttention, self).__init__()
		if config.hidden_size % config.num_attention_heads != 0:
			raise ValueError(
				"The hidden size (%d) is not a multiple of the number of attention "
				"heads (%d)" % (config.hidden_size, config.num_attention_heads))
		self.output_attentions = config.output_attentions

		self.num_attention_heads = config.num_attention_heads
		self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
		self.all_head_size = self.num_attention_heads * self.attention_head_size

		self.query = nn.Linear(config.hidden_size, self.all_head_size)
		self.key = nn.Linear(config.hidden_size, self.all_head_size)
		self.value = nn.Linear(config.hidden_size, self.all_head_size)

		self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

	def transpose_for_scores(self, x):
		new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
		x = x.view(*new_x_shape)
		return x.permute(0, 2, 1, 3)

	def forward(self, hidden_states, attention_mask, head_mask=None):
		mixed_query_layer = self.query(hidden_states)
		mixed_key_layer = self.key(hidden_states)
		mixed_value_layer = self.value(hidden_states)

		query_layer = self.transpose_for_scores(mixed_query_layer)
		key_layer = self.transpose_for_scores(mixed_key_layer)
		value_layer = self.transpose_for_scores(mixed_value_layer)

		# Take the dot product between "query" and "key" to get the raw attention scores.
		attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
		attention_scores = attention_scores / math.sqrt(self.attention_head_size)
		# Apply the attention mask is (precomputed for all layers in BertModel forward() function)
		attention_scores = attention_scores + attention_mask

		# Normalize the attention scores to probabilities.
		attention_probs = nn.Softmax(dim=-1)(attention_scores)

		# This is actually dropping out entire tokens to attend to, which might
		# seem a bit unusual, but is taken from the original Transformer paper.
		attention_probs = self.dropout(attention_probs)

		# Mask heads if we want to
		if head_mask is not None:
			attention_probs = attention_probs * head_mask

		context_layer = torch.matmul(attention_probs, value_layer)

		context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
		new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
		context_layer = context_layer.view(*new_context_layer_shape)

		outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
		return outputs


class BertSelfOutput(nn.Module):
	def __init__(self, config):
		super(BertSelfOutput, self).__init__()
		self.dense = nn.Linear(config.hidden_size, config.hidden_size)
		self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

	def forward(self, hidden_states, input_tensor):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.dropout(hidden_states)
		hidden_states = self.LayerNorm(hidden_states + input_tensor)
		return hidden_states


class BertAttention(nn.Module):
	def __init__(self, config):
		super(BertAttention, self).__init__()
		self.self = BertSelfAttention(config)
		self.output = BertSelfOutput(config)
		self.pruned_heads = set()

	def prune_heads(self, heads):
		if len(heads) == 0:
			return
		mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
		heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
		for head in heads:
			# Compute how many pruned heads are before the head and move the index accordingly
			head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
			mask[head] = 0
		mask = mask.view(-1).contiguous().eq(1)
		index = torch.arange(len(mask))[mask].long()

		# Prune linear layers
		self.self.query = prune_linear_layer(self.self.query, index)
		self.self.key = prune_linear_layer(self.self.key, index)
		self.self.value = prune_linear_layer(self.self.value, index)
		self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

		# Update hyper params and store pruned heads
		self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
		self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
		self.pruned_heads = self.pruned_heads.union(heads)

	def forward(self, input_tensor, attention_mask, head_mask=None):
		self_outputs = self.self(input_tensor, attention_mask, head_mask)
		attention_output = self.output(self_outputs[0], input_tensor)
		outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
		return outputs


class BertIntermediate(nn.Module):
	def __init__(self, config):
		super(BertIntermediate, self).__init__()
		self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
		if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
			self.intermediate_act_fn = ACT2FN[config.hidden_act]
		else:
			self.intermediate_act_fn = config.hidden_act

	def forward(self, hidden_states):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.intermediate_act_fn(hidden_states)
		return hidden_states


class BertOutput(nn.Module):
	def __init__(self, config):
		super(BertOutput, self).__init__()
		self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
		self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

	def forward(self, hidden_states, input_tensor):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.dropout(hidden_states)
		hidden_states = self.LayerNorm(hidden_states + input_tensor)
		return hidden_states


class BertLayer(nn.Module):
	def __init__(self, config):
		super(BertLayer, self).__init__()
		self.attention = BertAttention(config)
		self.intermediate = BertIntermediate(config)
		self.output = BertOutput(config)

	def forward(self, hidden_states, attention_mask, head_mask=None):
		attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
		attention_output = attention_outputs[0]
		intermediate_output = self.intermediate(attention_output)
		layer_output = self.output(intermediate_output, attention_output)
		outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
		return outputs

class SplitBertEncoder(nn.Module):
	def __init__(self, config):
		super(SplitBertEncoder, self).__init__()
		self.output_attentions = config.output_attentions
		self.output_hidden_states = config.output_hidden_states
		self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
		self.split = 999

	def redistribute(self, split):
		for i in range(len(self.layer)):
			if i >= split:
				self.layer[i].to('cuda:1')
		self.split = split

	def forward(self, hidden_states, attention_mask, head_mask=None):
		all_hidden_states = ()
		all_attentions = ()
		for i, layer_module in enumerate(self.layer):
			if self.output_hidden_states:
				all_hidden_states = all_hidden_states + (hidden_states,)

			if i == self.split:
				hidden_states = hidden_states.cuda(1)
				attention_mask = attention_mask.cuda(1)

			layer_outputs = layer_module(hidden_states, attention_mask)
			hidden_states = layer_outputs[0]

			if self.output_attentions:
				if i >= self.split:
					layer_outputs[1] = layer_outputs[1].cuda(0)
				all_attentions = all_attentions + (layer_outputs[1],)

		# Add last layer
		if self.output_hidden_states:
			all_hidden_states = all_hidden_states + (hidden_states,)
			for i in range(len(all_hidden_states)):
				if i >= self.split:
					all_hidden_states[i] = all_hidden_states[i].cuda(0)
		
		if self.split < len(self.layer):
			hidden_states = hidden_states.cuda(0)
		outputs = (hidden_states,)
		if self.output_hidden_states:
			outputs = outputs + (all_hidden_states,)
		if self.output_attentions:
			outputs = outputs + (all_attentions,)
		return outputs  # last-layer hidden state, (all hidden states), (all attentions)

class BertEmbeddings(nn.Module):
	"""Construct the embeddings from word, position and token_type embeddings.
	"""
	def __init__(self, config):
		super(BertEmbeddings, self).__init__()
		self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
		self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
		self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

		# self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
		# any TensorFlow checkpoint file
		self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

	def forward(self, input_ids, token_type_ids=None, position_ids=None):
		seq_length = input_ids.size(1)
		if position_ids is None:
			position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
			position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
		if token_type_ids is None:
			token_type_ids = torch.zeros_like(input_ids)

		words_embeddings = self.word_embeddings(input_ids)
		position_embeddings = self.position_embeddings(position_ids)
		token_type_embeddings = self.token_type_embeddings(token_type_ids)

		embeddings = words_embeddings + position_embeddings + token_type_embeddings
		embeddings = self.LayerNorm(embeddings)
		embeddings = self.dropout(embeddings)
		return embeddings

class BertPooler(nn.Module):
	def __init__(self, config):
		super(BertPooler, self).__init__()
		self.dense = nn.Linear(config.hidden_size, config.hidden_size)
		self.activation = nn.Tanh()

	def forward(self, hidden_states):
		# We "pool" the model by simply taking the hidden state corresponding
		# to the first token.
		first_token_tensor = hidden_states[:, 0]
		pooled_output = self.dense(first_token_tensor)
		pooled_output = self.activation(pooled_output)
		return pooled_output

class BertModel(BertPreTrainedModel):

	def __init__(self, config):
		super(BertModel, self).__init__(config)

		self.embeddings = BertEmbeddings(config)
		self.encoder = SplitBertEncoder(config)
		self.pooler = BertPooler(config)

		self.init_weights()

	def _resize_token_embeddings(self, new_num_tokens):
		old_embeddings = self.embeddings.word_embeddings
		new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
		self.embeddings.word_embeddings = new_embeddings
		return self.embeddings.word_embeddings

	def _resize_type_embeddings(self, new_num_types):
		old_embeddings = self.embeddings.token_type_embeddings
		new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_types)
		self.embeddings.token_type_embeddings = new_embeddings
		return self.embeddings.token_type_embeddings

	def _prune_heads(self, heads_to_prune):

		for layer, heads in heads_to_prune.items():
			self.encoder.layer[layer].attention.prune_heads(heads)

	def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None):
		if attention_mask is None:
			attention_mask = torch.ones_like(input_ids)
		if token_type_ids is None:
			token_type_ids = torch.zeros_like(input_ids)

		extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
		extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
		extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

		if head_mask is not None:
			if head_mask.dim() == 1:
				head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
				head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
			elif head_mask.dim() == 2:
				head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
			head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
		else:
			head_mask = [None] * self.config.num_hidden_layers
		
		if inputs_embeds is not None:
			embedding_output = inputs_embeds
		else:
			embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
		encoder_outputs = self.encoder(embedding_output,
									   extended_attention_mask,
									   head_mask=head_mask)
		sequence_output = encoder_outputs[0]
		pooled_output = self.pooler(sequence_output)

		outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
		return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

class RobertaEmbeddings(BertEmbeddings):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    def __init__(self, config):
        super(RobertaEmbeddings, self).__init__(config)
        self.padding_idx = 1

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            # Position numbers begin at padding_idx+1. Padding symbols are ignored.
            # cf. fairseq's `utils.make_positions`
            #position_ids = torch.arange(self.padding_idx+1, seq_length+self.padding_idx+1, dtype=torch.long, device=input_ids.device)
            #position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            mask = input_ids.ne(self.padding_idx).long()
            position_ids = torch.cumsum(mask, dim=1).type_as(mask) * mask + self.padding_idx
        return super(RobertaEmbeddings, self).forward(input_ids,
                                                      token_type_ids=token_type_ids,
                                                      position_ids=position_ids)

class RobertaModel(BertModel):

    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaModel, self).__init__(config)

        self.embeddings = RobertaEmbeddings(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None):
        if input_ids[:, 0].sum().item() != 0:
            logger.warning("A sequence with no special tokens has been passed to the RoBERTa model. "
                           "This model requires special tokens in order to work. "
                           "Please specify add_special_tokens=True in your encoding.")
        return super(RobertaModel, self).forward(input_ids,
                                                 attention_mask=attention_mask,
                                                 token_type_ids=token_type_ids,
                                                 position_ids=position_ids,
                                                 head_mask=head_mask, inputs_embeds=inputs_embeds)


class SequenceSummary(nn.Module):
	r""" Compute a single vector summary of a sequence hidden states according to various possibilities:
		Args of the config class:
			summary_type:
				- 'last' => [default] take the last token hidden state (like XLNet)
				- 'first' => take the first token hidden state (like Bert)
				- 'mean' => take the mean of all tokens hidden states
				- 'cls_index' => supply a Tensor of classification token position (GPT/GPT-2)
				- 'attn' => Not implemented now, use multi-head attention
			summary_use_proj: Add a projection after the vector extraction
			summary_proj_to_labels: If True, the projection outputs to config.num_labels classes (otherwise to hidden_size). Default: False.
			summary_activation: 'tanh' => add a tanh activation to the output, Other => no activation. Default
			summary_first_dropout: Add a dropout before the projection and activation
			summary_last_dropout: Add a dropout after the projection and activation
	"""
	def __init__(self, config):
		super(SequenceSummary, self).__init__()

		self.summary_type = config.summary_type if hasattr(config, 'summary_use_proj') else 'last'
		if self.summary_type == 'attn':
			# We should use a standard multi-head attention module with absolute positional embedding for that.
			# Cf. https://github.com/zihangdai/xlnet/blob/master/modeling.py#L253-L276
			# We can probably just use the multi-head attention module of PyTorch >=1.1.0
			raise NotImplementedError

		self.summary = Identity()
		if hasattr(config, 'summary_use_proj') and config.summary_use_proj:
			if hasattr(config, 'summary_proj_to_labels') and config.summary_proj_to_labels and config.num_labels > 0:
				num_classes = config.num_labels
			else:
				num_classes = config.hidden_size
			self.summary = nn.Linear(config.hidden_size, num_classes)

		self.activation = Identity()
		if hasattr(config, 'summary_activation') and config.summary_activation == 'tanh':
			self.activation = nn.Tanh()

		self.first_dropout = Identity()
		if hasattr(config, 'summary_first_dropout') and config.summary_first_dropout > 0:
			self.first_dropout = nn.Dropout(config.summary_first_dropout)

		self.last_dropout = Identity()
		if hasattr(config, 'summary_last_dropout') and config.summary_last_dropout > 0:
			self.last_dropout = nn.Dropout(config.summary_last_dropout)

	def forward(self, hidden_states, cls_index=None):
		""" hidden_states: float Tensor in shape [bsz, seq_len, hidden_size], the hidden-states of the last layer.
			cls_index: [optional] position of the classification token if summary_type == 'cls_index',
				shape (bsz,) or more generally (bsz, ...) where ... are optional leading dimensions of hidden_states.
				if summary_type == 'cls_index' and cls_index is None:
					we take the last token of the sequence as classification token
		"""
		if self.summary_type == 'last':
			output = hidden_states[:, -1]
		elif self.summary_type == 'first':
			output = hidden_states[:, 0]
		elif self.summary_type == 'mean':
			output = hidden_states.mean(dim=1)
		elif self.summary_type == 'cls_index':
			if cls_index is None:
				cls_index = torch.full_like(hidden_states[..., :1, :], hidden_states.shape[-2]-1, dtype=torch.long)
			else:
				cls_index = cls_index.unsqueeze(-1).unsqueeze(-1)
				cls_index = cls_index.expand((-1,) * (cls_index.dim()-1) + (hidden_states.size(-1),))
			# shape of cls_index: (bsz, XX, 1, hidden_size) where XX are optional leading dim of hidden_states
			output = hidden_states.gather(-2, cls_index).squeeze(-2) # shape (bsz, XX, hidden_size)
		elif self.summary_type == 'attn':
			raise NotImplementedError

		output = self.first_dropout(output)
		output = self.summary(output)
		output = self.activation(output)
		output = self.last_dropout(output)

		return output

class Trilinear_Att_layer(nn.Module):
	def __init__(self, config):
		super(Trilinear_Att_layer, self).__init__()
		self.W1 = nn.Linear(config.hidden_size, 1)
		self.W2 = nn.Linear(config.hidden_size, 1) 
		self.W3 = nn.Parameter(torch.Tensor(1, 1, config.hidden_size))
		init.kaiming_uniform_(self.W3, a=math.sqrt(5))

	def forward(self, u, u_mask, v, v_mask):
		part1 = self.W1(u)     # batch * seq_len * 1
		part2 = self.W2(v).permute(0, 2, 1)   # batch * 1 * seq_len
		part3 = torch.bmm(self.W3*u, v.permute(0, 2, 1))  # batch * seq_len * seq_len
		u_mask = (1.0 - u_mask.float()) * -10000.0
		v_mask = (1.0 - v_mask.float()) * -10000.0
		joint_mask = u_mask.unsqueeze(2) + v_mask.unsqueeze(1)    # batch * seq_len * num_paths
		total_part = part1 + part2 + part3 + joint_mask
		return total_part

class OCN_Att_layer(nn.Module):
	def __init__(self, config):
		super(OCN_Att_layer, self).__init__()
		self.att = Trilinear_Att_layer(config)

	def forward(self, ol, ol_mask, ok, ok_mask):
		#print ('ol', ol.shape)
		#print ('ok', ok.shape)
		A = self.att(ol, ol_mask, ok, ok_mask)
		att = F.softmax(A, dim=1)    
		_OLK = torch.bmm(ol.permute(0, 2, 1), att).permute(0, 2, 1)       # batch *  hidden * seq_len
		OLK = torch.cat([ok-_OLK, ok*_OLK], dim=2)
		return OLK

class OCN_CoAtt_layer(nn.Module):
	def __init__(self, config):
		super(OCN_CoAtt_layer, self).__init__()
		self.att = Trilinear_Att_layer(config)
		self.Wp = nn.Linear(config.hidden_size*3, config.hidden_size)

	def forward(self, d, d_mask, OCK, OCK_mask):
		A = self.att(d, d_mask, OCK, OCK_mask)
		ACK = F.softmax(A, dim=2)    
		OA = torch.bmm(ACK, OCK)   
		APK = F.softmax(A, dim=1)
		POAA = torch.bmm(torch.cat([d, OA], dim=2).permute(0, 2, 1), APK).permute(0, 2, 1)
		OPK = F.relu(self.Wp(torch.cat([OCK, POAA], dim=2)))
		return OPK

class OCN_SelfAtt_layer(nn.Module):
	def __init__(self, config):
		super(OCN_SelfAtt_layer, self).__init__()
		self.att = Trilinear_Att_layer(config)
		self.Wf = nn.Linear(config.hidden_size*4, config.hidden_size)

	def forward(self, OPK, OPK_mask, _OPK, _OPK_mask):
		A = self.att(OPK, OPK_mask, _OPK, _OPK_mask)
		att = F.softmax(A, dim=1)    
		OSK = torch.bmm(OPK.permute(0, 2, 1), att).permute(0, 2, 1)      
		OFK = torch.cat([_OPK, OSK, _OPK-OSK, _OPK*OSK], dim=2)
		OFK = F.relu(self.Wf(OFK))
		return OFK

class OCN_Merge_layer(nn.Module):
	def __init__(self, config):
		super(OCN_Merge_layer, self).__init__()
		self.Wc = nn.Linear(config.hidden_size*9, config.hidden_size)
		self.Va = nn.Linear(config.hidden_size, 1)
		self.Wg = nn.Linear(config.hidden_size*3, config.hidden_size)

	def forward(self, o1, o1o2, o1o3, o1o4, o1o5, q, q_mask):
		q_mask = (1.0 - q_mask.float()) * -10000.0
		Aq = F.softmax(self.Va(q)+q_mask.unsqueeze(2), dim=1)
		Q = torch.bmm(q.permute(0, 2, 1), Aq).permute(0, 2, 1).repeat(1, o1.shape[1], 1)
		OCK = torch.tanh(self.Wc(torch.cat([o1, o1o2, o1o3, o1o4, o1o5], dim=2)))   # batch * seq_len * hidden
		G = torch.sigmoid(self.Wg(torch.cat([o1, OCK, Q], dim=2)))
		out = G*o1 + (1-G)*OCK
		return out

class QANet_Att_layer(nn.Module):
	def __init__(self, config):
		super(QANet_Att_layer, self).__init__()
		self.commonsense_encoder = nn.LSTM(input_size=config.hidden_size, hidden_size=100, batch_first=True, bidirectional=True, dropout=config.hidden_dropout_prob)
		self.commonsense_projector = nn.Linear(200, config.hidden_size)
		self.W1 = nn.Linear(config.hidden_size, 1) # keep bias for now, need to check what zero_ is doing
		self.W2 = nn.Linear(config.hidden_size, 1) # keep bias for now, need to check what zero_ is doing
		self.W3 = nn.Parameter(torch.Tensor(1, 1, config.hidden_size))
		init.kaiming_uniform_(self.W3, a=math.sqrt(5))
		self.att_proj = nn.Linear(config.hidden_size*4, config.hidden_size)
		self.softmax = nn.Softmax(dim=2)
		self.sigmoid = nn.Sigmoid()
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

	def forward(self, bert_output, commonsense, attention_mask, commonsense_mask, commonsense_shape):
		b_size, num_cand, num_path, path_len = commonsense_shape
		lstm_output, (h, c) = self.commonsense_encoder(commonsense)
		encoded_cs = h.permute(1, 0, 2).contiguous().view(b_size*num_cand*num_path, -1)
		projected_cs = F.relu(self.commonsense_projector(encoded_cs).view(b_size*num_cand, num_path, -1))
		commonsense = projected_cs
		part1 = self.W1(bert_output)     # batch * seq_len * 1
		part2 = self.W2(commonsense).permute(0, 2, 1)   # batch * 1 * num_paths
		part3 = torch.bmm(self.W3*bert_output, commonsense.permute(0, 2, 1))  # batch * seq_len * num_paths
		mask_cs = (1.0 - commonsense_mask.float()) * -10000.0
		mask_dqa = (1.0 - attention_mask.float()) * -10000.0
		joint_mask = mask_cs.unsqueeze(1) + mask_dqa.unsqueeze(2)    # batch * seq_len * num_paths
		total_part = part1 + part2 + part3 + joint_mask
		cs_att = self.softmax(total_part)    
		weight_cs = torch.bmm(cs_att, commonsense)       # batch * seq_len * hidden
		context_att = F.softmax(total_part, dim=1).permute(0, 2, 1)
		probs = torch.bmm(cs_att, context_att)
		weighted_context = torch.bmm(probs, bert_output)
		cs_attended = F.relu(self.att_proj(torch.cat([bert_output, weight_cs, bert_output*weight_cs, bert_output*weighted_context], dim=2)))
		cs_attended = self.LayerNorm(cs_attended + bert_output)
		return cs_attended

class KVMem_Att_layer(nn.Module):
	def __init__(self, config):
		super(KVMem_Att_layer, self).__init__()
		self.commonsense_encoder = nn.LSTM(input_size=config.hidden_size, hidden_size=100, batch_first=True, bidirectional=True, dropout=config.hidden_dropout_prob)
		self.commonsense_wk = nn.Linear(200, config.hidden_size)
		self.commonsense_wv = nn.Linear(200, config.hidden_size)
		self.num_attention_heads = config.num_attention_heads
		self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
		self.all_head_size = self.num_attention_heads * self.attention_head_size
		self.att_dropout = nn.Dropout(config.attention_probs_dropout_prob)

		self.output = nn.Linear(config.hidden_size, config.hidden_size)
		self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

	def transpose_for_scores(self, x):
		new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
		x = x.view(*new_x_shape)
		return x.permute(0, 2, 1, 3)

	def forward(self, bert_output, commonsense, attention_mask, commonsense_mask, commonsense_shape):
		b_size, num_cand, num_path, path_len = commonsense_shape
		lstm_output, (h, c) = self.commonsense_encoder(commonsense)
		encoded_cs = h.permute(1, 0, 2).contiguous().view(b_size*num_cand*num_path, -1)
		cs_key = self.commonsense_wk(encoded_cs).view(b_size*num_cand, num_path, -1)
		cs_value = self.commonsense_wv(encoded_cs).view(b_size*num_cand, num_path, -1)

		query_layer = self.transpose_for_scores(bert_output)
		key_layer = self.transpose_for_scores(cs_key)
		value_layer = self.transpose_for_scores(cs_value)

		attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
		attention_scores = attention_scores / math.sqrt(self.attention_head_size)

		mask_cs = (1.0 - commonsense_mask.float()) * -10000.0
		mask_dqa = (1.0 - attention_mask.float()) * -10000.0
		joint_mask = mask_cs.unsqueeze(1) + mask_dqa.unsqueeze(2)    # batch * seq_len * num_paths
		joint_mask = joint_mask.unsqueeze(1)   # batch * 1 * seq_len * num_paths
		attention_scores = attention_scores + joint_mask
		attention_probs = nn.Softmax(dim=-1)(attention_scores)
		attention_probs = self.att_dropout(attention_probs)

		context_layer = torch.matmul(attention_probs, value_layer)

		context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
		new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
		context_layer = context_layer.view(*new_context_layer_shape)
		cs_attended = self.output(context_layer)
		cs_attended = self.dropout(cs_attended)
		cs_attended = self.LayerNorm(cs_attended + bert_output)
		return cs_attended
