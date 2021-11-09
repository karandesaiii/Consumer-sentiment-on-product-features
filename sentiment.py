import nltk
from nltk.corpus import stopwords
from nltk.parse.stanford import StanfordDependencyParser
import os
import networkx as nx
from textblob import TextBlob
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

# java_path = "C:/Program Files/Java/jre1.8.0_311/bin/java.exe"
os.environ['JAVAHOME'] = config['JAVA']['java_path']

path_to_jar = config['STANFORD']['jar_path']
path_to_models_jar = config['STANFORD']['models_jar_path']
dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
is_noun = lambda pos: pos[:2] == 'NN'





sent = "The look of this cloth is great, it's the material that is mediocre"
# sent = "Quality of the cloth is really bad"
# sent = "Poor quality of hardware is the major issue"
# sent = "UX of this app sucks"

print()
print("Consumer review :",sent)

stop_words = set(stopwords.words('english'))
tokenized = nltk.word_tokenize(sent)
tokenized = [w for w in tokenized if not w in stop_words]
sent = ""
for w in tokenized:
    sent += " " + w


idx_list = []
current_index = 0
for word in tokenized:
    idx_list.append(str(current_index))
    current_index += 1

#noun index
nouns = []
noun_idx = []
current_index = 0
for (word, pos) in nltk.pos_tag(tokenized):
    if is_noun(pos):
        nouns.append(word)
        noun_idx.append(str(current_index))
    current_index += 1


result = dependency_parser.raw_parse(sent)
dep = result.__next__()
dep_dot = dep.to_dot().split('\n')
edges = []
for entry in dep_dot:
    if entry.find('->') > 0:
        index_relation = entry.split(' ')[:3]
        edges.append((index_relation[0], index_relation[2]))

is_present = {}
for tupl in edges:
    for i in tupl:
        is_present[i] = 1

graph = nx.Graph(edges)

clusters = {}
for i in noun_idx:
    clusters[i] = []

idx_list = [w for w in idx_list if w in is_present]
noun_idx = [w for w in noun_idx if w in is_present]
for word in idx_list:
    k = -1
    min_dist = 100
    for target in noun_idx:
        d = nx.shortest_path_length(graph, source=word, target=target)
        if d < min_dist:
            min_dist = d
            k = target
    clusters[k].append(word)

theta = 3
for c1 in noun_idx:
    for c2 in noun_idx:
        if c1 != c2:
            d = nx.shortest_path_length(graph, source=c1, target=c2)
            if d < theta:
                clusters[c1] += clusters[c2]
                clusters[c2] = []


for key, value in clusters.items():
    sentence = ""
    if len(value) > 0 :
        for i in value:
           sentence += tokenized[int(i)] + " "
        print (sentence)
        result = TextBlob(sentence)
        score_final=result.sentiment.polarity
        print ("Sentiment score",score_final)
        if score_final<0:
            print("Sentiment is negative")
        elif score_final==0:
            print("Sentiment is neutral")
        else:
            print("Sentiment is positive")
        print()
