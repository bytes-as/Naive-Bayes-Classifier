import numpy as np

class BayesClassifier:
	def __init__(self):
		self._TRAIN_DATA_PATH=None
		self._TEST_DATA_PATH=None
		self.train_header=None
		self.train_data=None
		self.train_label=None
		self.train_classes=None
		self.class_freq=None
		self.test_header=None
		self.test_data=None
		self.test_label=None
		self.frequency_table=None

	def setTrainDataPath(self, PATH):
		self._TRAIN_DATA_PATH=PATH

	def setTestDataPath(self, PATH):
		self._TEST_DATA_PATH=PATH

	def loadTrainData(self):
		if 'csv' not in dir():
			import csv
		if self._TRAIN_DATA_PATH is None:
			raise Exception('No path is given for the train data')
		with open(self._TRAIN_DATA_PATH, 'rt') as readFile:
			data = csv.reader(readFile)
			header = next(data)[0].split(',')
			train_data = []
			for row in data:
				train_data.append([int(entry) for entry in row[0].split(',')])
		self.train_data = np.array(train_data)[:, 1:]
		self.train_label = np.array(train_data)[:, 0]
		self.train_header = np.array(header)[1:]
		self.train_decision_head = np.array(header)[0]
		print('[%-15s] : %-25s' %('Train Data', 'loaded successfully...'))

	def loadTestData(self):
		if 'csv' not in dir():
			import csv
		if self._TEST_DATA_PATH is None:
			raise Exception('No path is given for the train data')
		with open(self._TEST_DATA_PATH, 'rt') as readFile:
			data = csv.reader(readFile)
			header = next(data)[0].split(',')
			test_data = []
			for row in data:
				test_data.append([int(entry) for entry in row[0].split(',')])
		self.test_data = np.array(test_data)[:, 1:]
		self.test_label = np.array(test_data)[:, 0]
		self.test_header = np.array(header)[1:]
		self.test_decision_head = np.array(header)[0]
		print('[%-15s] : %-25s' %('Test Data', 'loaded successfully...'))

	def initializeFrequencyTable(self):
		if self.train_data is None:
			raise Exception('No train data found for classifier to learn')
		self.frequency_table = {}
		if 'numpy' not in dir():
			import numpy as np
		self.train_classes = np.unique(self.train_label)
		self.class_freq = {}
		for cl in self.train_classes:
			self.class_freq[cl]=0
		for index, attribute in enumerate(self.train_header):
			if attribute not in self.frequency_table:
				self.frequency_table[attribute] = np.zeros((5,len(self.train_classes)))
		print('[%-15s] : %-25s' %('Frequency Table', 'Initialized...'))

	def learnFrequencyTable(self):
		if self.frequency_table is None:
			self.initializeFrequencyTable()
		for i, row in enumerate(self.train_data):
			self.class_freq[self.train_label[i]] += 1
			label_index = np.where(self.train_classes == self.train_label[i])[0]
			for j, attribute_value in enumerate(row):
				self.frequency_table[self.train_header[j]][attribute_value-1, label_index] += 1
		print('[%-15s] : %-25s' %('Frequency Table', 'parcing complete...'))


	def printFrequencyTable(self, attribute):
		if self.frequency_table is None:
			raise Exception('First make the classifier learn things\n Use clf.learnFrequencyTable()')
		if attribute in self.train_classes:
			raise Exception('given argument is a class not an attribute')
		if attribute not in self.train_header:
			raise Exception('Key Error: \'%s\' is not a valid attribute', attribute)
		print('\n[%-10s] : %-12s'%('attr = '+attribute,'------labels------'))
		print('%-12s : [%-6s]  [%-6s]'%('attr values',self.train_classes[0], self.train_classes[1]))
		for index, row in enumerate(self.frequency_table[attribute]):
			print('[%-10d] : %-8d  %-8d' %(
				index+1,
				row[0], row[1]
				))
		print('-------X----------------X-------')

	def getPrediction(self, data_point):
		probabilities = {}
		for i, label in enumerate(self.train_classes):
			prob = self.class_freq[0]/(self.class_freq[0] + self.class_freq[1])
			for j, attr_val in enumerate(data_point):
				attribute = self.train_header[j]
				p_numerator = (self.frequency_table[attribute][attr_val-1, i]+1)/(self.class_freq[i]+6)
				# p_denominator = ((self.frequency_table[attribute][attr_val-1, i]+1)/(self.class_freq[i])+6) + \
					# ((self.frequency_table[attribute][attr_val-1, 1-i]+1)/(self.class_freq[1-i])+6)
				prob *= p_numerator
			probabilities[label] = prob
		return max(probabilities, key=lambda k: probabilities[k])

	def evaluateClassifier(self,on='train', test_data_path=None):
		if on=='test':
			print('[%-15s] : %-25s' %('Evaluation', 'on test data...'))
			if test_data_path is not None:
				self.setTestDataPath(test_data_path)
				self.loadTestData()
			if self.test_data is None:
				if self._TEST_DATA_PATH is None:
					raise Exception('Insert the TEST DATA PATH!!!')
				else:
					self.loadTestData()
			right = 0
			wrong = 0
			for i, row in enumerate(self.test_data):
				if self.test_label[i] == self.getPrediction(row):	right += 1
				else:	wrong += 1
			print('[%-15s] : %-25s' %('Evaluation', '[[TEST]] --> Accuracy = ' + str((right/(right+wrong)) * 100) + '%'))
		elif on=='train':
			print('[%-15s] : %-25s' %('Evaluation', 'on train data...'))
			if self.train_data is None:
				raise Exception('Insert the TRAIN DATA and make the model learn!!!')
			if self.frequency_table is None:
				raise Exception('Fit the data first!!!')
			right = wrong = 0
			for i, row in enumerate(self.train_data):
				if self.train_label[i] == self.getPrediction(row):	right += 1
				else:	wrong += 1
			print('[%-15s] : %-25s' %('Evaluation', '[[TRAIN]] --> Accuracy = ' + str((right/(right+wrong)) * 100) + '%'))
		else:
			raise Exception("invlid argument %s, valid argument are 'test' and 'train'"%on)

	def fit(self, train_data_path=None):
		if train_data_path is not None:
			self.setTrainDataPath(train_data_path)
			self.loadTrainData()
		if self.train_data is None:
			if self._TRAIN_DATA_PATH is None:
				raise Exception('set the train data path')
			else:
				self.loadTrainData()
		self.initializeFrequencyTable()
		self.learnFrequencyTable()
		self.evaluateClassifier()

if __name__=='__main__':
	clf = BayesClassifier()
	clf.setTrainDataPath('./data2_19.csv')
	clf.setTestDataPath('./test2_19.csv')
	clf.loadTrainData()
	clf.initializeFrequencyTable()
	clf.learnFrequencyTable()
	''' to print the frequency tables of the attributes based on the train data'''
	for attr in clf.train_header:
		clf.printFrequencyTable(attr)
	clf.loadTestData()
	clf.evaluateClassifier()
	clf.evaluateClassifier(on='test')
	
	
	
	'''              or       '''
	print('\n')
	clf.fit(train_data_path='./data2_19.csv')
	clf.evaluateClassifier(on='test', test_data_path='./test2_19.csv')