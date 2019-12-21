import numpy as np
import csv

class BayesClassifier:
	def __init__(self):
		self._TRAIN_DATA_PATH=None
		self._TEST_DATA_PATH=None

		self.train_header=None
		self.train_data=None
		self.train_classes=None
		self.train_unique_values={}

		self.test_header=None
		self.test_data=None

		self.seperated_data=None
		self.frequency_table=None
		self.class_freq=None

	def setTrainDataPath(self, PATH):
		self._TRAIN_DATA_PATH=PATH

	def setTestDataPath(self, PATH):
		self._TEST_DATA_PATH=PATH

	def loadTrainData(self):
		if 'csv' not in dir():
			import csv
		if self._TRAIN_DATA_PATH is None:
			raise Exception('No path is given for the train data')
		if self.train_data is None:
			self.train_data = []
		else:
			raise Exception('there is train data already in the classifier')
		with open(self._TRAIN_DATA_PATH, 'rt') as readFile:
			data = csv.reader(readFile)
			header = next(data)[0].split(',')
			for row in data:
				self.train_data.append([int(entry) for entry in row[0].split(',')])
		self.train_data = np.array(self.train_data)
		self.train_header = np.array(header)
		print('[%-15s] : %-25s' %('Train Data', 'loaded successfully...'))

	def loadTestData(self):
		if 'csv' not in dir():
			import csv
		if self._TEST_DATA_PATH is None:
			raise Exception('No path is given for the test data')
		if self.test_data is None:
			self.test_data = []
		else:
			raise Exception('there is test data in the clalssifer already!!!')
		with open(self._TEST_DATA_PATH, 'rt') as readFile:
			data = csv.reader(readFile)
			header = next(data)[0].split(',')
			for row in data:
				self.test_data.append([int(entry) for entry in row[0].split(',')])
		self.test_data = np.array(self.test_data)
		self.test_header = np.array(header)
		print('[%-15s] : %-25s' %('Test Data', 'loaded successfully...'))

	def seperateByClass(self, data='train'):
		seperated_data = {}
		if data=='train':
			for row in self.train_data:
				if row[0] not in seperated_data:
					seperated_data[row[0]] = [row]
				else:
					seperated_data[row[0]].append(row)
		else:
			for row in self.test_data:
				if row[0] not in seperated_data:
					seperated_data[row[0]] = [row]
				else:
					seperated_data[row[0]].append(row)
		self.seperated_data=seperated_data

	def initializeFrequencyTable(self):
		if self.train_data is None:
			raise Exception('No train data found for classifier to learn')
		self.frequency_table = {}
		if 'numpy' not in dir():
			import numpy as np
		self.train_classes = np.unique(self.train_data[:,0])
		self.class_freq = {}
		for cl in self.train_classes:
			self.class_freq[cl]=0
		# print('train classes: ', self.train_classes)
		for index, attribute in enumerate(self.train_header[1:]):
			self.train_unique_values[attribute] = np.unique(self.train_data[:,index+1])
			# print(attribute, self.train_unique_values[attribute])
			if attribute not in self.frequency_table:
				self.frequency_table[attribute] = np.zeros((
					len(self.train_unique_values[attribute]),
					len(self.train_classes)
				))
		# for attribute in self.train_header[1:]:
		# 	attribute_index = np.where(self.train_header == attribute)[]
		# 	unique_values_for_attribute = np.unique()
		print('[%-15s] : %-25s' %('Frequency Table', 'Initialized successfully...'))

	def learnFrequencyTable(self):
		if self.frequency_table is None:
			self.initializeFrequencyTable()
		for row in self.train_data:
			self.class_freq[row[0]] += 1
			label_index = np.where(self.train_classes == row[0])[0]
			for index, attribute_value in enumerate(row[1:]):
				attribute_value_index = np.where(self.train_unique_values[self.train_header[index+1]] == attribute_value)[0]
				# print(self.frequency_table[self.train_header[index+1]][attribute_value_index, label_index])
				self.frequency_table[self.train_header[index+1]][attribute_value_index, label_index] += 1
		print('[%-15s] : %-25s' %('Frequency Table', 'Generated from the train data...'))


	def printFrequencyTable(self, attribute):
		if self.frequency_table is None:
			raise Exception('First make the classifier learn things\n Use clf.learnFrequencyTable()')
		if attribute in self.train_classes:
			raise Exception('given argument is a class not an attribute')
		if attribute not in self.train_header:
			raise Exception('Key Error: \'%s\' is not an attribute', attribute)
		print('\n[%-10s] : %-12s'%('attr = '+attribute,'------labels------'))
		print('%-12s : [%-6s]  [%-6s]'%('attr values',self.train_classes[0], self.train_classes[1]))
		for index, row in enumerate(self.frequency_table[attribute]):
			print('[%-10d] : %-8d  %-8d' %(
				self.train_unique_values[attribute][index],
				row[0], row[1]
				))
		print('-------X----------------X-------')

	def getPrediction(self, data_point):
		label = self.train_classes[0]
		final_prob = self.class_freq[0]/(self.class_freq[0] + self.class_freq[1])
		for i, attr_val in enumerate(data_point[1:]):
			attr = self.test_header[i+1]
			label_index = np.where(self.train_classes == label)[0][0]
			if label_index==0: non_label = 1
			else: non_label=0
			attr_value_index = np.where(self.train_unique_values[attr] == attr_val)[0]
			require_prob = (self.frequency_table[attr][attr_value_index,label_index] + 1)/(self.class_freq[label_index] + 7)
			attr_total_prob = ((self.frequency_table[attr][attr_value_index,label_index]+1)/(self.class_freq[label_index])+7) + \
				((self.frequency_table[attr][attr_value_index,non_label]+1)/(self.class_freq[non_label])+7)
			final_prob *= require_prob/attr_total_prob
		prob1 = final_prob
		
		label = self.train_classes[1]
		final_prob = self.class_freq[1]/(self.class_freq[0] + self.class_freq[1])
		for i, attr_val in enumerate(data_point[1:]):
			attr = self.test_header[i+1]
			label_index = np.where(self.train_classes == label)[0][0]
			if label_index==0: non_label = 1
			else: non_label=0
			attr_value_index = np.where(self.train_unique_values[attr] == attr_val)[0]
			require_prob = (self.frequency_table[attr][attr_value_index,label_index] + 1)/(self.class_freq[label_index] + 7)
			attr_total_prob = ((self.frequency_table[attr][attr_value_index,label_index]+1)/(self.class_freq[label_index])+7) + \
				((self.frequency_table[attr][attr_value_index,non_label]+1)/(self.class_freq[non_label])+7)
			final_prob *= require_prob/attr_total_prob
		prob2 = final_prob
		if prob1 <= prob2:
			return self.train_classes[1]
		else:
			return self.train_classes[0]

	def evaluateClassifier(self):
		print('[%-15s] : %-25s' %('Evaluation', 'Starting evaluation process...'))
		if self.test_data is None:
			raise Exception('Insert the TEST DATA!!!')
		right = 0
		wrong = 0
		for row in self.test_data:
			if row[0] == self.getPrediction(row):
				right += 1
			else:
				wrong += 1
		# print('right:', right, '    wrong:', wrong)
		print('[%-15s] : %-25s' %('Evaluation', 'Accuracy = ' + str((right/(right+wrong)) * 100) + '%'))

if __name__=='__main__':
	clf = BayesClassifier()
	clf.setTrainDataPath('./data2_19.csv')
	# clf.setTrainDataPath('./test2_19.csv')
	clf.loadTrainData()
	clf.setTestDataPath('./test2_19.csv')
	clf.loadTestData()
	clf.initializeFrequencyTable()
	clf.learnFrequencyTable()
	''' to print the frequency tables of the attributes based on the train data'''
	# for attr in clf.train_header[1:]:
	# 	clf.printFrequencyTable(attr)
	clf.evaluateClassifier()