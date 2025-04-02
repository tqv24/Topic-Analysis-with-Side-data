import argparse
from train_LDA_NN import train_LDANN 
#changed!!!!
from data_nnlda import DataPreProcess
import pickle

parser = argparse.ArgumentParser(description='Topic modeling')
parser.add_argument('--config_path', type=str, default='config_nn', help='input parameter directory')
parser.add_argument('--data_path', type=str, default='data', help='input data directory')
parser.add_argument('--test_data_path', type=str, default='test_data', help='testing data directory')
parser.add_argument('--number_of_topic_group', type=int, default=5, help='number of topic group')
args = parser.parse_args()

train_data = DataPreProcess(args.data_path, args.config_path)
model = train_LDANN(args.number_of_topic_group, train_data.word_id, train_data.count_voc, train_data.text_matrix, len(train_data.voc))
model.train_model(train_data.config['--num_iter'], float(train_data.config['--lr']),train_data.config['--NN_num_iter'], train_data.config['--prior_from_p_LDA'])
model.getTopWords(train_data.config['--top_num'], train_data.config['--top_word_file'], train_data.id2w)
model.getTopicID(train_data.config['--topic_id_added_data'], train_data.id2s, train_data.id2w)
model.rating_perf(train_data.rating, train_data.id2s, train_data.id2w, train_data.config['--top_num'])
fileObject = open('LDANN_model', 'wb')
fileObject1 = open('Data_model_nn', 'wb')
pickle.dump(model, fileObject)
pickle.dump(train_data, fileObject1)
train_data.get_topic_word(model.id_doc,args.number_of_topic_group,5)
#model1 = pickle.load(open('LDANN_model','rb'))
#train_data1 = pickle.load(open('Data_model_nn','rb'))
#train_data1.pro_test_file(args.test_data_path)
#pred = model1.lda_nn_inf(train_data1.test_word_id, train_data1.test_count_voc, train_data1.test_text_matrix, train_data1.id2w, train_data1.test_text_raw)
#print(pred)
