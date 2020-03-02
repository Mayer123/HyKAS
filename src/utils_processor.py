import json 
import os 
import logging
import numpy as np
from commonsense_mapping import COMMONSENSE_MAPPING
from collections import Counter
from utils import DataProcessor, InputExample, answerKey_mapping
import csv

logger = logging.getLogger(__name__)

class PIQAProcessor(DataProcessor):
	def __init__(self, data_dir):
		self.D = [[], [], []]
		label_files = [os.path.join(data_dir, 'train-labels.lst'), os.path.join(data_dir, 'valid-labels.lst')]
		for sid in range(2):
			with open([os.path.join(data_dir, "train.jsonl"), os.path.join(data_dir, "valid.jsonl")][sid], "r") as f:
				data = []
				for line in f:
					data.append(json.loads(line))
				labels = self.read_labels(label_files[sid])
				for i in range(len(data)):
					d = ['Q: ' + data[i]["goal"]]
					d += ['A: ' + data[i]['sol1']]
					d += ['A: ' + data[i]['sol2']]
					d += [labels[i]] 
					self.D[sid] += [d]
		
	def read_labels(self, file):
		labels = []
		with open(file, 'r') as f:
			for line in f:
				labels.append(line.strip())
		return labels

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
		return ["0", "1"]

	def _create_examples(self, data, set_type):
		"""Creates examples for the training and dev sets."""
		examples = []
		for (i, d) in enumerate(data):
			answer = str(data[i][-1])		

			for k in range(2):
				guid = "%s-%s-%s" % (set_type, i, k)
				text_b = data[i][k+1]
				text_a = data[i][0]
				examples.append(
						InputExample(guid=guid, text_a=text_a, text_b=text_b, label=answer))
			
		return examples

class SocialIQAProcessor(DataProcessor):
	def __init__(self, data_dir):
		self.D = [[], [], []]
		for sid in range(2):
			with open([os.path.join(data_dir, "socialIQa_v1.4_trn.jsonl"), os.path.join(data_dir, "socialIQa_v1.4_dev.jsonl")][sid], "r") as f:
				data = []
				for line in f:
					data.append(json.loads(line))
				for i in range(len(data)):
					d = ['D: ' + data[i]['context']]
					d += ['Q: ' + data[i]["question"]]
					d += ['A: ' + data[i]['answerA']]
					d += ['A: ' + data[i]['answerB']]
					d += ['A: ' + data[i]['answerC']]
					d += [answerKey_mapping[data[i]['correct']]] 
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
		return ["0", "1", "2"]

	def _create_examples(self, data, set_type):
		"""Creates examples for the training and dev sets."""
		examples = []
		for (i, d) in enumerate(data):
			answer = str(data[i][-1])		

			for k in range(3):
				guid = "%s-%s-%s" % (set_type, i, k)
				text_b = data[i][k+2]
				text_a = data[i][1]
				text_c = data[i][0]
				examples.append(
						InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, label=answer))
			
		return examples

class CosmosQAProcessor(DataProcessor):
	def __init__(self, data_dir):
		self.D = [[], [], []]
		for sid in range(2):
			with open([os.path.join(data_dir, "train.jsonl"), os.path.join(data_dir, "valid.jsonl")][sid], "r") as f:
				data = []
				for line in f:
					data.append(json.loads(line))
				for i in range(len(data)):
					d = ['D: ' + data[i]['context']]
					d += ['Q: ' + data[i]["question"]]
					d += ['A: ' + data[i]['answer0']]
					d += ['A: ' + data[i]['answer1']]
					d += ['A: ' + data[i]['answer2']]
					d += ['A: ' + data[i]['answer3']]
					d += [data[i]['label']] 
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
		return ["0", "1", "2", "3"]

	def _create_examples(self, data, set_type):
		"""Creates examples for the training and dev sets."""
		examples = []
		for (i, d) in enumerate(data):
			answer = str(data[i][-1])		

			for k in range(4):
				guid = "%s-%s-%s" % (set_type, i, k)
				text_b = data[i][k+2]
				text_a = data[i][1]
				text_c = data[i][0]
				examples.append(
						InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, label=answer))
			
		return examples

class SwagProcessor(DataProcessor):
	def __init__(self, data_dir):
		self.D = [[], [], []]
		for sid in range(2):
			data = self._read_csv([os.path.join(data_dir, "train.csv"), os.path.join(data_dir, "val.csv")][sid])
			for i in range(len(data)):
				if i == 0:
					continue
				d = [data[i][4]]   # context
				d += [data[i][5]]  # question
				if data[i][5].find("_") != -1:
					print (data[i][5], 'fuck')
					exit(0)
				d += [data[i][7]]  # A1 
				d += [data[i][8]]  # A2 
				d += [data[i][9]]  # A3 
				d += [data[i][10]] # A4 
				d += [data[i][11]]  # label
				self.D[sid] += [d]
	
	def _read_csv(self, input_file):
		with open(input_file, "r", encoding="utf-8") as f:
			return list(csv.reader(f))

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
		return ["0", "1", "2", "3"]

	def _create_examples(self, data, set_type):
		"""Creates examples for the training and dev sets."""
		examples = []
		for (i, d) in enumerate(data):
			answer = str(data[i][-1])		

			for k in range(4):
				guid = "%s-%s-%s" % (set_type, i, k)
				text_b = data[i][k+2]
				text_a = data[i][1]
				text_c = data[i][0]
				examples.append(
						InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, label=answer))
			
		return examples

newprocessors = {
	"piqa": PIQAProcessor, # max_length 150
	"socialiqa": SocialIQAProcessor, # max_length 90
	"cosmosqa": CosmosQAProcessor, # max_length 200
	"swag": SwagProcessor, # max_length 100
}

newoutput_modes = {
	"piqa": "classification",
	"socialiqa": "classification",
	"cosmosqa": "classification",
	"swag": "classification"
}