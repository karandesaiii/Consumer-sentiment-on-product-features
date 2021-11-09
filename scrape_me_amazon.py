import bs4 as bs
import urllib.request, traceback
import nltk
from nltk.corpus import stopwords
from nltk.parse.stanford import StanfordDependencyParser
import os
import networkx as nx
from textblob import TextBlob
java_path = "C:/Program Files/Java/jre1.8.0_311/bin/java.exe"
os.environ['JAVAHOME'] = java_path

path_to_jar = 'C:/Users/Arya/Downloads/Karan/Projects/Amazon-review-summarization-master/stanford-corenlp-3.8.0.jar'
path_to_models_jar = 'C:/Users/Arya/Downloads/Karan/Projects/Amazon-review-summarization-master/stanford-corenlp-3.8.0-models.jar'
dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
is_noun = lambda pos: pos[:2] == 'NN'



class Comment_Analysis:
    def __init__(self, link):
        self.link = link

    def print_all_comments(self):
        # reaching the comment section
        sauce = urllib.request.urlopen(self.link).read()
        soup = bs.BeautifulSoup(sauce, 'lxml')
        body = soup.find('body')
        div = body.find('div')
        div = div.find('div', {'id':'dp'})
        div = div.find('div', class_='a-container')
        div = div.find('div', class_='a-row a-spacing-extra-large')
        div = div.find('div', class_='a-column a-span8')
        div = div.find('div', {'id':'cr-medley-top-reviews-wrapper'})
        div = div.find('div', class_='a-row')
        div = div.find('div', class_='a-section reviews-content filterable-reviews-content celwidget')
        div = div.find('div')
        comment_div_list = div.find_all('div', class_='a-section review')
        # passing each user's comment section to a funtion to extract just the comment as text
        for d in comment_div_list:
            self.print_comment_text(d)
        print('\n------------------------------------------------------------------------\n\n')


    def print_comment_text(self, d):
        # getting the comment text from HTML comment block
        comment_div = d.find('div')
        comment_div = comment_div.find('div', class_="a-row review-data")
        comment_span = comment_div.find('span')
        comment_div = comment_span.find('div')
        comment_div = comment_div.find('div', class_="a-expander-content a-expander-partial-collapse-content")
        comment_text = comment_div.text
        print('\n\n\n',comment_text,'\n')
        #tokenizing into sentences and passing each sentence for sentiment analysis
        sentences = nltk.sent_tokenize(comment_text)
        for sent in sentences:
            self.analyse(sent)


    def analyse(self, sent):
        stop_words = set(stopwords.words('english'))
        tokenized = nltk.word_tokenize(sent)
        #removing stop words
        tokenized = [w for w in tokenized if not w in stop_words]
        sent = "" #new sentence without stopwords
        for w in tokenized:
            sent += " " + w

        #creating a word-position-index.
        #This will help to create graph even in the presence of duplicate nouns.
        idx_list = []
        current_index = 0
        for word in tokenized:
            idx_list.append(str(current_index))
            current_index += 1

        #nouns and noun index
        nouns = []
        noun_idx = []
        current_index = 0
        for (word, pos) in nltk.pos_tag(tokenized):
            if is_noun(pos):
                nouns.append(word)
                noun_idx.append(str(current_index))
            current_index += 1

        #making a dependency tree using stanford dependency parser
        #creating a graph using the dependencies
        result = dependency_parser.raw_parse(sent)
        dep = result.__next__()
        dep_dot = dep.to_dot().split('\n')
        edges = [] #this will save all the edges. (Connection between two 2 nodes)
        for entry in dep_dot:
            if entry.find('->') > 0:
                index_relation = entry.split(' ')[:3]
                edges.append((index_relation[0], index_relation[2]))
        graph = nx.Graph(edges)

        #making a list of all the present nodes in the graph
        is_present = {}
        for tupl in edges:
            for i in tupl:
                is_present[i] = 1

        # defining cluster for each feature
        clusters = {}
        for i in noun_idx:
            clusters[i] = []

        # distributing the words into the cluster they are closest to
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

        # merging clusters if their mutual distance is less than a threshold - theta
        # this is done to group the nouns which are defining similar things
        theta = 3
        for c1 in noun_idx:
            for c2 in noun_idx:
                if c1 != c2:
                    d = nx.shortest_path_length(graph, source=c1, target=c2)
                    if d < theta:
                        clusters[c1] += clusters[c2]
                        clusters[c2] = []

        # getting the sentence for each cluster (feature) for sentiment analysis
        for key, value in clusters.items():
            sentence = ""
            if len(value) > 0 :
                for i in value:
                   sentence += tokenized[int(i)] + " "
                print (sentence)
                # doing sentiment analysis
                result = TextBlob(sentence)
                print (result.sentiment)



if __name__ == "__main__":
    inferno_link = "https://www.amazon.in/Inferno-Robert-Langdon-Dan-Brown/dp/0552169595/ref=pd_cp_14_2?_encoding=UTF8&psc=1&refRID=4GS6S8WZHGD8GXJ00DX8"
    lg_phone = "https://www.amazon.in/LG-Q6-Black-18-FullVision/dp/B074H1PCLG/ref=br_asw_pdt-4?pf_rd_m=A1VBAL9TL5WCBF&pf_rd_s=&pf_rd_r=MMTPZ5JKKZ6X8H0W1HQA&pf_rd_t=36701&pf_rd_p=e3173096-0c04-4a84-a379-e54a50298b47&pf_rd_i=desktop"
    laptop = "https://www.amazon.in/gp/product/B07568CCSV/ref=s9u_simh_gw_i3?ie=UTF8&pd_rd_i=B07568CCSV&pd_rd_r=8c7bfb31-ef35-11e7-ad7e-9b193ab508f2&pd_rd_w=apH2s&pd_rd_wg=0deye&pf_rd_m=A1VBAL9TL5WCBF&pf_rd_s=&pf_rd_r=MMTPZ5JKKZ6X8H0W1HQA&pf_rd_t=36701&pf_rd_p=a66bc199-b270-44de-9fcc-5cf0a06a7727&pf_rd_i=desktop"
    cmt = Comment_Analysis(inferno_link)
    cmt.print_all_comments()

#    pos_sent = "I have an ipod and it is a great buy but I'm probably the only person that dislikes the iTunes software."
#    neg_sent = "worst product ever ordered. System hangs a lot. lost my critical reports due to hanging issue. not at all worth purchasing."
#    cmt.analyse(neg_sent)
