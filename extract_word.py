from nltk import word_tokenize, pos_tag, ne_chunk
from nltk import RegexpParser
from nltk import Tree
import pandas as pd
import nltk, re, pprint

def ie_preprocess(document):
	sentences = nltk.sent_tokenize(document)
	sentences = [nltk.word_tokenize(sent) for sent in sentences]
	sentences = [nltk.pos_tag(sent) for sent in sentences]
	return sentences

def get_continuous_chunks(text, chunk_func=ne_chunk):
	grammar = {}
	grammar[0] = "NP: {<DT|JJ|NN.*>+}"
	grammar[1] = "PP: {<IN><NP>}"
	grammar[2] = "VP: {<VB.*><NP|PP|CLAUSE>+$}"
	grammar[3] = "CLAUSE: {<NP><VP>}"
	continuous_chunk_total = []
	for i in range(4):
		cp = nltk.RegexpParser(grammar[i])
		sentences = ie_preprocess(text)
		chunked = cp.parse(sentences[0])
		continuous_chunk = []
		current_chunk = []

		for subtree in chunked:
			if type(subtree) == Tree:
				current_chunk.append(" ".join([token for token, pos in subtree.leaves()]))
			elif current_chunk:
				named_entity = " ".join(current_chunk)
				if named_entity not in continuous_chunk:
					continuous_chunk.append(named_entity)
					current_chunk = []
			else:
				continue
		for i in range(len(continuous_chunk)):
			continuous_chunk_total.append(continuous_chunk[i])
	return continuous_chunk_total

sent = 'Make better quality products that work. Less diaper leaks. '
result = get_continuous_chunks(sent)
print(result)