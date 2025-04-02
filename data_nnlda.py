import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.utils import shuffle
import string
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.preprocessing import OneHotEncoder
import random
import enchant
import re

#random missing categorical info
#root od every word

class DataPreProcess:
	def __init__(self, data_file_direc, config_file_direc='config_nn'):
		def load_para_from_file(config_path):
			param = {}
			with open(config_path) as f:
				for line in f:
					name, value = line.strip().split("\t")
					param[name] = value
			return param
		self.config = load_para_from_file( config_file_direc  )
		self.le = []
		self.enc = None
		def decontracted(phrase):
			# specific
			phrase = re.sub(r"won\'t", "will not", phrase)
			phrase = re.sub(r"can\'t", "can not", phrase)

			# general
			phrase = re.sub(r"n\'t", "not", phrase)
			phrase = re.sub(r"\'re", " are", phrase)
			phrase = re.sub(r"\'s", " is", phrase)
			phrase = re.sub(r"\'d", " would", phrase)
			phrase = re.sub(r"\'ll", " will", phrase)
			phrase = re.sub(r"\'t", " not", phrase)
			phrase = re.sub(r"\'ve", " have", phrase)
			phrase = re.sub(r"\'m", " am", phrase)
			return phrase
		def read_data(stop_words, data_file_direc, rating_name, comment_name, category_number):
			category_number = int(category_number)
			table = pd.read_table(data_file_direc, sep=",")
			stemmer = SnowballStemmer("english")
			d = enchant.Dict('en_US')
			text_origin = []
			for it in (list(table)[0:7]):
				text_origin.append(table[it].values.tolist())
			text_origin = np.array(text_origin)
			text_origin = np.transpose(text_origin)
			num_fea = text_origin.shape[1]
			num_remove = int(num_fea/2)
			ran = list(np.arange(0,num_fea))
			for i in range(text_origin.shape[0]):
				index = random.sample(ran, num_remove)
				for j in range(num_remove):
					text_origin[i,index[j]] = 'null'

			text = []
			for it in range(category_number,category_number+1):
				temp = preprocessing.LabelEncoder()
				temp.fit(text_origin[:,it])
				self.le.append(temp)
				text.append(list(temp.transform(list(text_origin[:,it]))))

			#old version of labelencoder without null block
			#for it in (list(table)[5:6]):
			#	temp = preprocessing.LabelEncoder()
			#	temp.fit(list(table[it]))
			#	self.le.append(temp)
			#	text.append(list(temp.transform(list(table[it]))))

			
			#le.fit(list(table[' sector']))
			#text.append(list(le.transform(list(table[' sector']))))
			text = np.array(text)
			text_matrix = np.transpose(text)
			self.enc = OneHotEncoder(handle_unknown='ignore')
			self.enc.fit(text_matrix)
			#text_matrix_part = self.enc.transform(text_matrix).toarray()
			text_matrix = self.enc.transform(text_matrix).toarray()
			#text_day = []
			#text_day.append(list(table['day']))
			#test_day = np.array(text_day)
			#text_day = np.transpose(test_day)
			#day_min = min(text_day[:,0])
			#day_max = max(text_day[:,0])
			#day = np.round_((text_day[:,0] - day_min)/(day_max-day_min))
			#text_matrix = np.zeros((day.shape[0], text_matrix_part.shape[1] + 1))
			#text_matrix[:,0] = day
			#text_matrix[:,1:text_matrix_part.shape[1] + 1] = text_matrix_part

			doc_num = text_matrix.shape[0]

			rating_raw = list(table[rating_name])

			delete_word = set(stopwords.words('english'))
			translator = str.maketrans('','',string.punctuation)
			for it in stop_words:
				delete_word.add(it)

			data_raw = list(table[comment_name])
			for i,it in enumerate(data_raw):
				data_raw[i] = decontracted(str(it))
			data_raw = np.array([data_raw])
			data_raw = np.transpose(data_raw)

			word_need_fix = {}
			for line in data_raw:
				sentence = list(str(line).lower().translate(translator).split())
				for i in range(len(sentence)):
					word = sentence[i]
					if word not in delete_word:
						root = stemmer.stem(word)
						if not d.check(root):
							if root not in word_need_fix:
								word_need_fix[root] = {}
								word_need_fix[root][word] = 1
							else:
								if word not in word_need_fix[root]:
									word_need_fix[root][word] = 1
								else:
									word_need_fix[root][word] +=1
			word_deter = {}
			for it in word_need_fix:
				temp_word = it
				temp_count = 0
				for it_s in word_need_fix[it]:
					if word_need_fix[it][it_s] > temp_count:
						temp_word = it_s
				word_deter[it] = temp_word

			data = []
			for line in data_raw:
				sentence = str(line).lower().translate(translator).split()
				new_sentence = []
				for i in range(len(sentence)):
					word = sentence[i]
					if word not in delete_word:
						root = stemmer.stem(word)
						if root in word_deter:
							new_sentence.append(word_deter[root])
						else:
							new_sentence.append(root)
				data.append(new_sentence)


			voc = {}
			id2s = {}
			count = 0
			count_s = 0
			for line in data:
				id2s[count_s] = line
				count_s += 1
				for token in line:
					if token in delete_word:
						continue
					if token not in voc :
						voc[token] = count
						count += 1
			data = np.array(data)
			data = np.transpose(data)
			#print(data)
			id2w = {voc[it]:it for it in voc}
			count_matrix = np.zeros((data.shape[0], len(voc)))
			for i,line in enumerate(data):
					for token in line:
						if token not in delete_word:
							count_matrix[i,voc[token]] += 1
			count_matrix_mod = []
			for i in range(count_matrix.shape[0]):
				row = []
				for j in range(count_matrix.shape[1]):
					row.append((j,count_matrix[i,j]))
				count_matrix_mod.append(row)
			count_matrix_final = []
			for i in range(count_matrix.shape[0]):
				bow = [it for it in count_matrix_mod[i] if it[1]!= 0 ]
				count_matrix_final.append(bow)
			#print(id2s)
			return rating_raw, count_matrix_final, text_matrix, voc, id2w, id2s, doc_num, text_origin	
		self.rating, self.count_matrix_r, self.text_matrix, self.voc, self.id2w, self.id2s, self.doc_num, self.text_origin = read_data(self.config["--stop_words"].split(","),data_file_direc,self.config["--rating_name"], self.config["--comment_name"], self.config["--category_number"])
		self.word_id = [   [w[0]   for w in it] for it in self.count_matrix_r]	
		self.count_voc = [[w[1] for w in it ] for it in self.count_matrix_r]
		file_3 = open('missing_text_file', 'w')
		for i in range(len(self.id2s)):
			file_3.write(str(self.text_origin[i][int(self.config["--category_number"])])+'\n')
		file_3.close()
	def get_topic_word(self, topic_id, top_num, number_words):
		file = open('bigram_description_of_topic_group','w')
		delete_word = set(stopwords.words('english'))
		translator = str.maketrans('','',string.punctuation)
		stop_words = self.config["--stop_words"].split(",")
		for it in stop_words:
			delete_word.add(it)

		str_doc = {}
		top_num = int(top_num)
		for i in range(top_num):
			str_doc[i] = []
		for i in range(topic_id.shape[0]):
			str_doc[topic_id[i]].append([self.id2s[i]])

		voc2 = {}
		id2w2 = {}
		data_2 = {}
		for i in range(top_num):
			voc2[i] = {}
			data_2[i] = []
			count_w = 0
			id2w2 = {}
			str_doc[i] = np.array(str_doc[i])
			for j,line in enumerate(str_doc[i]):
				count_2w = 0
				sample = []
				for w in line:
					for token in w: 
						if token in delete_word:
							continue
						if count_2w == 0:
							word_1 = token
							count_2w = 1
						else:
							word_2 = word_1
							word_2 = word_2 + ' ' + token
							sample.append(word_2)
							if word_2 not in voc2[i]:
								voc2[i][word_2] = count_w
								count_w += 1
							word_1 = token
				data_2[i].append(sample)
			data_2[i] = np.array(data_2[i])
			id2w2[i] = {voc2[i][it]:it for it in voc2[i]}
			count_matrix2 = np.zeros((data_2[i].shape[0], len(voc2[i])))
			#print(len(voc2[i]))

			for k, line in enumerate(data_2[i]):
				for w in line:
					count_matrix2[k,voc2[i][w]] += 1
			word_count = np.sum(count_matrix2, axis=0)
			word_index = word_count.argsort()[-number_words:][::-1]
			file.write('topic_group_' + str(i) + '\t')
			for k in range(number_words):
				if id2w2[i] == {}:
					break
				file.write(str(id2w2[i][word_index[k]])+'\t')
			file.write('\n')
		file.close()
	def pro_test_file(self, test_file_direct):
		def decontracted(phrase):
			# specific
			phrase = re.sub(r"won\'t", "will not", phrase)
			phrase = re.sub(r"can\'t", "can not", phrase)

			# general
			phrase = re.sub(r"n\'t", "not", phrase)
			phrase = re.sub(r"\'re", " are", phrase)
			phrase = re.sub(r"\'s", " is", phrase)
			phrase = re.sub(r"\'d", " would", phrase)
			phrase = re.sub(r"\'ll", " will", phrase)
			phrase = re.sub(r"\'t", " not", phrase)
			phrase = re.sub(r"\'ve", " have", phrase)
			phrase = re.sub(r"\'m", " am", phrase)
			return phrase
		def read_data(stop_words, data_file_direc):
			table = pd.read_table(data_file_direc, sep=",")
			stemmer = SnowballStemmer("english")
			d = enchant.Dict('en_US')
			text = []
			text_raw = []
			#text.append(list(table['day']))
			count = 0
			for it in (list(table)[6:7]):
				text_raw.append(list(table[it]))
				text.append(list(self.le[count].transform(list(table[it]))))
				count += 1
			#le.fit(list(table[' sector']))
			#text.append(list(le.transform(list(table[' sector']))))
			text_raw = np.array(text_raw)
			text_raw = np.transpose(text_raw)
			text = np.array(text)
			text_matrix = np.transpose(text)
			#text_matrix_part = self.enc.transform(text_matrix).toarray()
			text_matrix = self.enc.transform(text_matrix).toarray()
			#doc_num = text_matrix_part.shape[0]
			#text_day = []
			#text_day.append(list(table['day']))
			#test_day = np.array(text_day)
			#text_day = np.transpose(test_day)
			#day_min = min(text_day[:,0])
			#day_max = max(text_day[:,0])
			#day = np.round_((text_day[:,0] - day_min)/(day_max-day_min))
			#text_matrix = np.zeros((doc_num, text_matrix_part.shape[1] + 1))
			#text_matrix[:,0] = day
			#text_matrix[:,1:] = text_matrix_part


			delete_word = set(stopwords.words('english'))
			translator = str.maketrans('','',string.punctuation)
			for it in stop_words:
				delete_word.add(it)

			data_raw = list(table['voc'])
			for i,it in enumerate(data_raw):
				data_raw[i] = decontracted(str(it))
			data_raw = np.array([data_raw])
			data_raw = np.transpose(data_raw)

			word_need_fix = {}
			for line in data_raw:
				sentence = list(str(line).lower().translate(translator).split())
				for i in range(len(sentence)):
					word = sentence[i]
					root = stemmer.stem(word)
					if not d.check(root):
						if root not in word_need_fix:
							word_need_fix[root] = {}
							word_need_fix[root][word] = 1
						else:
							if word not in word_need_fix[root]:
								word_need_fix[root][word] = 1
							else:
								word_need_fix[root][word] +=1
			word_deter = {}
			for it in word_need_fix:
				temp_word = it
				temp_count = 0
				for it_s in word_need_fix[it]:
					if word_need_fix[it][it_s] > temp_count:
						temp_word = it_s
				word_deter[it] = temp_word

			data = []
			count_s = 0
			id2s = {}
			for line in data_raw:
				sentence = str(line).lower().translate(translator).split()
				for i in range(len(sentence)):
					word = sentence[i]
					root = stemmer.stem(word)
					if root in word_deter:
						sentence[i] = word_deter[root]
					else:
						sentence[i] = root
				data.append(sentence)
				id2s[count_s] = sentence
				count_s += 1

			data = np.array(data)
			data = np.transpose(data)
			#id2w = {voc[it]:it for it in voc}
			count_matrix = np.zeros((data.shape[0], len(self.voc)))
			for i,line in enumerate(data):
					for token in line:
						token = stemmer.stem(token)
						if token not in delete_word:
							count_matrix[i,self.voc[token]] += 1
			count_matrix_mod = []
			for i in range(count_matrix.shape[0]):
				row = []
				for j in range(count_matrix.shape[1]):
					row.append((j,count_matrix[i,j]))
				count_matrix_mod.append(row)
			count_matrix_shrink = []
			for i in range(count_matrix.shape[0]):
				bow = [it for it in count_matrix_mod[i] if it[1]!= 0 ]
				count_matrix_shrink.append(bow)
			text_matrix_final = []
			count_matrix_final = []
			#rating_final = []
			id2s_f = {}
			count_id2s = 0
			for i in range(count_matrix.shape[0]):
				if count_matrix_shrink[i] != []:
					count_matrix_final.append(count_matrix_shrink[i])
					text_matrix_final.append(text_matrix[i])
					#rating_final.append(rating_raw[i])
					id2s_f[count_id2s] = id2s[i]
					count_id2s += 1
			text_matrix_final = np.array(text_matrix_final)
			return count_matrix_final, text_matrix_final , id2s_f, doc_num, text_raw
		self.test_count_matrix_r, self.test_text_matrix, self.test_id2s, self.test_doc_num, self.test_text_raw = read_data(self.config["--stop_words"].split(","),test_file_direct)
		self.test_word_id = [   [w[0]   for w in it] for it in self.test_count_matrix_r]	
		self.test_count_voc = [[w[1] for w in it ] for it in self.test_count_matrix_r]










