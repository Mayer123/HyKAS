import json 
import os 
import logging
import numpy as np
from commonsense_mapping import COMMONSENSE_MAPPING
from collections import Counter

logger = logging.getLogger(__name__)
answerKey_mapping = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}

def accuracy(out, labels):
	return {'acc': (out == labels).mean()}

class InputExample(object):

	def __init__(self, guid, text_a, text_b=None, concepts=None, label=None):
		self.guid = guid
		self.text_a = text_a
		self.text_b = text_b
		self.label = label
		self.concepts = concepts

class InputFeatures(object):
	"""A single set of features of data."""

	def __init__(self, guid, input_ids, input_mask, token_type_ids, label_id, concepts=None, concepts_mask=None, concepts_mask_full=None):
		self.guid = guid
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.token_type_ids = token_type_ids
		self.label_id = label_id
		self.concepts = concepts
		self.concepts_mask = concepts_mask
		self.concepts_mask_full = concepts_mask_full

def convert_examples_to_features(examples, tokenizer, max_length=512,
								 label_list=None,
								 output_mode=None,
								 pad_on_left=False,
								 pad_token=0,
								 pad_token_segment_id=0,
								 mask_padding_with_zero=True,
								 max_path=0, max_path_len=0, cls_token=0, sep_token=2):
	""" Loads a data file into a list of `InputBatch`s
		`cls_token_at_end` define the location of the CLS token:
			- False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
			- True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
		`cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
	"""

	label_map = {label : i for i, label in enumerate(label_list)}
	features = [[]]
	for (ex_index, example) in enumerate(examples):
		if ex_index % 10000 == 0:
			logger.info("Writing example %d of %d" % (ex_index, len(examples)))

		inputs = tokenizer.encode_plus(example.text_b, add_special_tokens=True, max_length=max_length)
		try:
			inputs_a = tokenizer.encode_plus(example.text_a, add_special_tokens=True, max_length=max_length-len(inputs['input_ids'])+1)
			inputs['input_ids'] = inputs_a['input_ids']+inputs['input_ids'][1:]
			inputs['token_type_ids'] = inputs_a['token_type_ids']+[1]*(len(inputs['token_type_ids'])-1)
		except:
			inputs_a = tokenizer.encode_plus(example.text_a, add_special_tokens=True, max_length=max_length)
			tmp_a, tmp_b, _ = tokenizer.truncate_sequences(inputs_a['input_ids'][:-1], pair_ids=inputs['input_ids'][1:-1], num_tokens_to_remove=len(inputs_a['input_ids'])+len(inputs['input_ids'])-max_length-1)
			inputs['input_ids'] = tmp_a+[sep_token]+tmp_b+[sep_token]
			inputs['token_type_ids'] = [0]*(len(tmp_a)+1)+[1]*(len(tmp_b)+1)
		concepts = None
		concepts_mask = None
		concepts_mask_full = None
		if example.concepts != None:
			concepts = []
			concepts_mask = []
			concepts_mask_full = []
			flat_list = []
			for c in example.concepts:
				cs_tokens = []
				cs_tokens.extend(tokenizer.tokenize(c[0].replace('_', ' ')))
				rel =  COMMONSENSE_MAPPING[c[1]]
				cs_tokens.extend(tokenizer.tokenize(rel, add_prefix_space=True))
				cs_tokens.extend(tokenizer.tokenize(c[2].replace('_', ' '), add_prefix_space=True))
				if len(c) > 3:
					rel = COMMONSENSE_MAPPING[c[3]]
					cs_tokens.extend(tokenizer.tokenize(rel, add_prefix_space=True))
					cs_tokens.extend(tokenizer.tokenize(c[4].replace('_', ' '), add_prefix_space=True))
				if cs_tokens in flat_list:
					continue
				if len(cs_tokens) > max_path_len:
					cs_tokens = cs_tokens[:max_path_len]
				flat_list.append(cs_tokens)
				concept_ids = tokenizer.convert_tokens_to_ids(cs_tokens)
				concept_ids = [cls_token] + concept_ids + [sep_token]
				concept_path_mask = [1]*len(concept_ids)
				while len(concept_ids) < max_path_len+2:
					concept_ids.append(pad_token)
					concept_path_mask.append(0)
				concepts.append(concept_ids)
				concepts_mask_full.append(concept_path_mask)
				concepts_mask.append(1)
				if len(concepts) == max_path:
					break
			while len(concepts) < max_path:
				concepts.append([cls_token, sep_token]+[pad_token]*max_path_len)
				concepts_mask_full.append([1, 1]+[0]*max_path_len)
				concepts_mask.append(0)

		input_ids, token_type_ids = inputs['input_ids'], inputs['token_type_ids']

		# The mask has 1 for real tokens and 0 for padding tokens. Only real
		# tokens are attended to.
		input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

		# Zero-pad up to the sequence length.
		padding_length = max_length - len(input_ids)
		if pad_on_left:
			input_ids = ([pad_token] * padding_length) + input_ids
			input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
			token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
		else:
			input_ids = input_ids + ([pad_token] * padding_length)
			input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
			token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

		assert len(input_ids) == max_length
		assert len(input_mask) == max_length
		assert len(token_type_ids) == max_length

		if example.label != None:
			if output_mode == "classification":
				label_id = label_map[example.label]
			elif output_mode == "regression":
				label_id = float(example.label)
			else:
				raise KeyError(output_mode)
		else:
			label_id = None

		if ex_index < 5:
			logger.info("*** Example ***")
			logger.info("guid: %s" % (example.guid))
			tokens = tokenizer.convert_ids_to_tokens(input_ids)
			logger.info("tokens: %s" % " ".join(
					[str(x) for x in tokens]))
			logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
			logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
			logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
			if example.label != None:
				logger.info("label: %s (id = %d)" % (example.label, label_id))
			if example.concepts:
				logger.info("commonsense_ids: %s" % " ".join([' '.join([str(xx) for xx in x]) for x in concepts]))
				logger.info("commonsense_masks_full: %s" % " ".join([' '.join([str(xx) for xx in x]) for x in concepts_mask_full]))
				logger.info("commonsense_masks: %s" % " ".join([str(x) for x in concepts_mask]))
				logger.info("commonsense tokens: %s" % " ".join([' '.join([str(xx) for xx in tokenizer.convert_ids_to_tokens(x)]) for x in concepts]))
		features[-1].append(
				InputFeatures(guid=example.guid, input_ids=input_ids,
							  input_mask=input_mask,
							  token_type_ids=token_type_ids,
							  label_id=label_id,
							  concepts=concepts, 
							  concepts_mask=concepts_mask,
							  concepts_mask_full=concepts_mask_full))
		if len(features[-1]) == len(label_list):
			features.append([])
	if len(features[-1]) == 0:
		features = features[:-1]
	return features

class DataProcessor(object):
	"""Base class for data converters for sequence classification data sets."""

	def get_train_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for the train set."""
		raise NotImplementedError()

	def get_dev_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for the dev set."""
		raise NotImplementedError()

	def get_labels(self):
		"""Gets the list of labels for this data set."""
		raise NotImplementedError()

	@classmethod
	def _read_tsv(cls, input_file, quotechar=None):
		"""Reads a tab separated value file."""
		with open(input_file, "r") as f:
			reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
			lines = []
			for line in reader:
				lines.append(line)
			return lines

class CommonsenseqaProcessor(DataProcessor):
	def __init__(self, data_dir):
		self.D = [[], [], []]
		for sid in range(2):
			with open([os.path.join(data_dir, "train_rand_split.jsonl"), os.path.join(data_dir, "dev_rand_split.jsonl")][sid], "r") as f:
				data = []
				for line in f:
					data.append(json.loads(line))
				for i in range(len(data)):
					d = ['Q: ' + data[i]["question"]['stem']]
					for k in range(len(data[i]['question']["choices"])):
						d += ['A: ' + data[i]['question']["choices"][k]['text']]
					d += [answerKey_mapping[data[i]["answerKey"]]] 
					self.D[sid] += [d]
		
	def get_train_examples(self, data_dir):
		"""See base class."""
		return self._create_examples(
				self.D[0], "train")

	def get_dev_examples(self, data_dir):
		"""See base class."""
		return self._create_examples(
				self.D[1], "dev")

	def get_labels(self):
		"""See base class."""
		return ["0", "1", "2", "3", "4"]

	def _create_examples(self, data, set_type):
		"""Creates examples for the training and dev sets."""
		examples = []
		for (i, d) in enumerate(data):
			answer = str(data[i][-1])		

			for k in range(5):
				guid = "%s-%s-%s" % (set_type, i, k)
				text_b = data[i][k+1]
				text_a = data[i][0]
				examples.append(
						InputExample(guid=guid, text_a=text_a, text_b=text_b, label=answer))
			
		return examples

class CommonsenseqaInjProcessor(DataProcessor):
	def __init__(self, data_dir):
		self.D = [[], [], []]
		len_dict = Counter()
		for sid in range(2):
			with open([os.path.join(data_dir, "train_cs.jsonl"), os.path.join(data_dir, "dev_cs.jsonl")][sid], "r") as f:
				data = []
				for line in f:
					data.append(json.loads(line))
				for i in range(len(data)):
					d = [data[i]['id'], 'Q: ' + data[i]["question"]['stem']]
					for k in range(len(data[i]['question']["choices"])):
						d += ['A: ' + data[i]['question']["choices"][k]['text']]
						d += [data[i]['choice_commonsense'][k]]
						len_dict[len(d[-1])] += 1
					d += [answerKey_mapping[data[i]["answerKey"]]] 
					self.D[sid] += [d]
		
	def get_train_examples(self, data_dir):
		"""See base class."""
		return self._create_examples(
				self.D[0], "train")

	def get_test_examples(self, data_dir):
		"""See base class."""
		examples = []
		with open(os.path.join(data_dir, "test_cs.jsonl"), 'r') as f:
			data = []
			for line in f:
				data.append(json.loads(line))
			for i in range(len(data)):
				question = 'Q: ' + data[i]['question']['stem']
				for k in range(5):
					guid = "%s-%s-%s" % ('test', i, k)
					candidate = 'A: '+data[i]['question']['choices'][k]['text']
					examples.append(InputExample(guid=guid, text_a=question, text_b=candidate, concepts=data[i]['choice_commonsense'][k])) 
		return examples

	def get_dev_examples(self, data_dir):
		"""See base class."""
		return self._create_examples(
				self.D[1], "dev")

	def get_labels(self):
		"""See base class."""
		return ["0", "1", "2", "3", "4"]

	def _create_examples(self, data, set_type):
		"""Creates examples for the training and dev sets."""
		examples = []
		for (i, d) in enumerate(data):
			answer = str(data[i][-1])		

			for k in range(5):
				guid = "%s-%s" % (data[i][0], k)
				text_b = data[i][k*2+2]
				text_a = data[i][1]
				examples.append(
						InputExample(guid=guid, text_a=text_a, text_b=text_b, concepts=data[i][k*2+3], label=answer))
			
		return examples

myprocessors = {
	"csqa": CommonsenseqaProcessor,
	"csqa-inj": CommonsenseqaInjProcessor
}

output_modes = {
	"csqa": "classification",
	"csqa-inj": "classification"
}