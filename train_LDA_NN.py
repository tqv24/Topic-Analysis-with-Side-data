import NN_lda
import torch
import numpy as np 
from NN_lda import Net
from onlineldavb import OnlineLDA
from scipy.special import gammaln, psi
import string
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from scipy.optimize import minimize
import pickle

class train_LDANN:
	def __init__(self, topic_num,word_id,word_count, text_matrix, len_voc):
		self.alpha_param = None
		self.beta_param = None
		self.topic_num = topic_num
		self.word_id = word_id
		self.word_count = word_count
		self.text_matrix = text_matrix
		self.len_voc = len_voc
		self.topic_doc = None
		self.word_topic = None
		self.lda_model = None
		self.nn_model = Net(text_matrix.shape[1], topic_num)
	def convert_lda_nn(self, input):
		input_tensor = torch.FloatTensor(input)
		return input_tensor
	def train_model(self, num_iter, lr, nn_num_iter, prior_from_p_lda):
		fileObject = open('LDA_model_in_LDANN', 'wb')
		fileObject1 = open('NN_model_in_LDANN', 'wb')
		k = self.topic_num
		prior_from_p_lda = np.array([float(it) for it in prior_from_p_lda.split(',')])
		text_matrix = self.convert_lda_nn(self.text_matrix)
		num_doc = self.text_matrix.shape[0]
		self.alpha_param = np.ones((len(self.word_id) ,k)) * (1.0/k)
		self.lda_model = OnlineLDA( k, self.len_voc, len(self.word_id), self.alpha_param, 0.01, 1.5, 0.75)
		for i in range(1):
			print(i)
			_,bound = self.lda_model.update_lambda(self.word_id, self.word_count, self.alpha_param)
			print(bound)
		self.topic_doc = self.lda_model.phi
		self.word_topic = self.lda_model._expElogbeta
		def loss_nn(alpha_param, topic_num, num_doc, wordids, wordcts, gamma,Elogbeta,eta ,lam,W):
			batchD = len(wordids)
			score = torch.FloatTensor([0])
			Elogtheta = torch.digamma(gamma) - torch.digamma(torch.sum(gamma, 1)).unsqueeze(1)
			expElogtheta = torch.exp(Elogtheta)

			# E[log p(docs | theta, beta)]
			for d in range(0, batchD):
				gammad = gamma[d, :]
				ids = wordids[d]
				cts = np.array(wordcts[d])
				phinorm = np.zeros(len(ids))
				for i in range(0, len(ids)):
					temp = Elogtheta[d, :] + Elogbeta[:, ids[i]]
					tmax = max(temp)
					phinorm[i] = np.log(sum(np.exp(temp - tmax))) + tmax
				score += np.sum(cts * phinorm)

			 # E[log p(theta | alpha) - log q(theta | gamma)]
			for it in range(num_doc):
				score += torch.sum((alpha_param[it] - gamma)*Elogtheta)
				score += torch.sum(torch.lgamma(gamma) - torch.lgamma(alpha_param[it]))
				score += torch.sum(torch.lgamma(alpha_param[it]*topic_num) - torch.lgamma(sum(gamma)))
			score = score/num_doc
 
			# Compensate for the subsampling of the population of documents
			score = score * num_doc / len(wordids)

			# E[log p(beta | eta) - log q (beta | lambda)]
			score = score + torch.sum((eta-lam)*Elogbeta)
			score = score + torch.sum(torch.lgamma(lam) - gammaln(eta))
			score = score + torch.sum(gammaln(eta*W) - torch.lgamma(torch.sum(lam, 1)))

			return(score)

		def loss_start(prior_from_p_lda,k):
			num_class = text_matrix.shape[1]
			text_flag = np.zeros((num_class, num_class))
			loss = 0.0
			for i in range(num_class):
				text_flag[i][i] = 1
			text_flag = self.convert_lda_nn(text_flag)
			mse = torch.nn.MSELoss()
			epochs = 20

			optimizer = torch.optim.Adam(self.nn_model.parameters(), lr = 0.001,weight_decay=0.1)
			label = self.convert_lda_nn(prior_from_p_lda).expand(text_flag.size(0),k)

			self.nn_model.train()

			for i in range(epochs):

				output = self.nn_model(text_flag)
				loss = mse(output,label)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

		loss_start(prior_from_p_lda,k)

		optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=lr, weight_decay = 0.1)
		self.nn_model.train()
		file_bound = np.zeros(int(num_iter))
		for i in range(int(num_iter)):
			print('NN work')
			print(i)
			phi = self.lda_model.phi
			phi = self.convert_lda_nn(phi)
			for j in range(int(nn_num_iter)):
				print(j)
				output = self.nn_model(text_matrix) + 1e-100
				loss = loss_nn(self.convert_lda_nn(output), k, num_doc, self.word_id, self.word_count, 
					self.convert_lda_nn(self.lda_model._gamma),self.convert_lda_nn(self.lda_model._Elogbeta),0.01, 
					self.convert_lda_nn(self.lda_model._lambda),self.len_voc) * (-1.0)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				#_,bound_nn = self.lda_model.update_lambda(self.word_id, self.word_count, self.nn_model(text_matrix).data.numpy() + 1e-100)
			self.alpha_param = output.data.numpy()
			#self.alpha_param = self.nn_model(text_matrix).data.numpy()
			file_bound[i] = -loss.item()
			#_, file_bound[i] = self.lda_model.update_lambda(self.word_id, self.word_count, self.alpha_param)
			print(file_bound[i])
			#20!!!!!
			for j in range(20):
				print(j)
				_,bound_lda = self.lda_model.update_lambda(self.word_id, self.word_count, self.alpha_param)
				print(bound_lda)
			pickle.dump(self.lda_model, fileObject)
			pickle.dump(self.nn_model, fileObject1)
		self.topic_doc = self.lda_model.phi
		self.word_topic = self.lda_model._expElogbeta
		np.save('bound', file_bound)
		print('perplexity' + '\n')
		print(bound_lda/(sum([np.sum(it) for it in self.word_count])))
		np.save('sd_alpha', self.alpha_param)
		#with open('trained_model.pt', 'wb') as f:
		#	torch.save(self.nn_model, f)
			#self.lda_model.trainLDA(count_matrix, self.alpha_param)

	def getTopWords(self, top_num, top_word_file, id2w):
		norm = self.word_topic
		k = self.topic_num
		np.save('topic_word_doc', norm)
		sorted_id = np.argsort(norm*(-1), axis=1)
		top_word = sorted_id[:,0:int(top_num)]
		file = open(top_word_file + str(k), "w")
		top_word_list = []
		for i in range(k):
			temp = [ id2w[it] for it in list(top_word[i,:])]
			top_word_list.append(temp)
			file.write('__topic__' + str(i+1) + '\t')
			for token in temp:
				file.write(token+' ')
			file.write('\n')
		file.close()
	def getTopicID(self, topic_id_added_data, id2s, id2w):
		file1 = open(topic_id_added_data, "w")
		file2 = open('word_predict_per_doc','w')
		norm = self.word_topic
		norm_word = norm
		norm_word = norm_word/np.sum(norm_word,axis=1)[:,np.newaxis]
		prob_matrix = self.topic_doc
		num_gen_word = 6
		alpha_para = self.alpha_param
		self.id_doc = np.argmax(prob_matrix,axis=1)
		k = self.topic_num
		for i in range(len(id2s)):
			if max(prob_matrix[i] - 1.0/k) <= 1e-6:
				file1.write('__topic__0' + '\t' + str(prob_matrix[i]) + '\t')
			else:
				index = np.argmax(prob_matrix[i,:]) + 1
				file1.write('__topic__'+str(index) + '\t' + str(prob_matrix[i]) + '\t')
			file1.write(str(id2s[i]) + '\n')
			prob_test = np.sum(norm_word*prob_matrix[i][:,np.newaxis],axis=0)
			max_index = prob_test.argsort()[-num_gen_word:][::-1]
			if prob_test[max_index[0]] == 0:
				prob_test = np.sum(norm_word*alpha_para[i][:,np.newaxis],axis=0)
				max_index = prob_test.argsort()[-num_gen_word:][::-1]
			#file2.write(id2s[i] + '\t')
			#use alpha_param!!!!!
			for s in range(num_gen_word):
				ind_index = max_index[s]
				if prob_test[ind_index] >= 1e-3:
					file2.write(id2w[ind_index] + ' ')
			file2.write('\t' + str(id2s[i]) + '\n')
		file1.close()
		file2.close()
	def rating_perf(self, rating, id2s, id2w, top_num_r):
		translator = str.maketrans('','',string.punctuation)
		top_num = int(top_num_r)

		k = self.topic_num
		norm = self.word_topic
		prob_matrix_r = np.zeros((self.text_matrix.shape[0], k + 1))
		prob_matrix_r[:,-1] = np.array(rating)
		sorted_id = np.argsort(norm*(-1), axis=1)
		top_word = sorted_id[:,0:top_num]

		top_word_list = []
		for i in range(k):
			temp = [ id2w[it] for it in list(top_word[i,:])]
			top_word_list.append(temp)

		w2p = {}
		for i in range(k):
			w2p[i] = {}

		for j in range(top_num):
			for i in range(k):
				col_id = sorted_id[i][j]
				w2p[i][top_word_list[i][j]] = norm[i][col_id]

		for i in range(self.text_matrix.shape[0]):
			line = id2s[i]
			for token in str(line).lower().translate(translator).split():
				for j in range(k):
					if token in w2p[j]:
						prob_matrix_r[i][j] = prob_matrix_r[i][j] + w2p[j][token]

		print("rating test")
		boost = xgb.sklearn.XGBClassifier()
		param = {'subsample':[0.8],'gamma':[0.0],'min_child_weight':[0],'max_depth': [3], 'learning_rate': [0.01],'objective' : ['multi:softprob'],"n_estimators":[50]}
		cvresult = GridSearchCV(boost,param,cv=10, verbose = 1, scoring = 'f1_micro')
		cvresult.fit(prob_matrix_r[:,:-1],prob_matrix_r[:,-1]-1)
		print(cvresult.best_score_)

	def lda_nn_inf(self, test_wordids, test_wordcts, test_textural_matrix,id2w, test_text_raw):
		alpha = self.nn_model(self.convert_lda_nn(test_textural_matrix)).data.numpy()
		_, _, doc_test = self.lda_model.inference(test_wordids,test_wordcts, alpha)
		test_topic_group = np.argmax(doc_test, axis=1)
		file_pred_word = open('predict_word', 'w')
		thro = 1e-2
		for i in range(doc_test.shape[0]):
			norm_doc = self.word_topic*(doc_test[i])[:,np.newaxis]
			file_pred_word.write(str(test_text_raw[i]) + '\t')
			for j in range(self.len_voc):
				for k in range(self.topic_num):
					if norm_doc[k][j] >= thro:
						file_pred_word.write(str(id2w[j]) + ' ')
			file_pred_word.write('\n')
		file_pred_word.close()
		return doc_test











