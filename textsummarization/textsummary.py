#!/usr/bin/env python
# coding: utf-8

# In[46]:


from newspaper import Article # newspaper : url에서 text를 크롤링 하는 패키지
from konlpy.tag import Kkma # KoNLPY : 한국 형태소 분석기 , KKMA : 문장 단위 분리 
from konlpy.tag import Twitter # Twitter : 명사 추출 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

import numpy as np


# In[47]:


class SentenceTokenizer(object):
    def __init__(self):
        self.kkma = Kkma()
        self.twitter = Twitter()
        self.stopwords = ['중인' ,'만큼', '마찬가지', '꼬집었', "연합뉴스", "데일리", "동아일보", "중앙일보", "조선일보", "기자"
                          ,"아", "휴", "아이구", "아이쿠", "아이고", "어", "나", "우리", "저희", "따라", "의해", "을", "를", "에", "의", "가",
                         "JTBC", ]
        
    # url 주소를 받아 기사내용을 추출  후 sentence를 return 
    def url2sentences(self, url):
        article = Article(url, language='ko')
        article.download()
        article.parse()
        sentences = self.kkma.sentences(article.text)
        
        for idx in range(0, len(sentences)):
            if len(sentences[idx]) <= 10:
                sentences[idx-1] += (' ' + sentences[idx])
                sentences[idx] = ''

        return sentences
    
    # text를 입력받아 setences를 return 
    def text2sentences(self, text):
        sentences = self.kkma.sentences(text)
        for idx in range(0, len(sentences)):
            if len(sentences[idx]) <= 10:
                sentences[idx-1] += (' ' + sentences[idx])
                sentences[idx] = ''
                
        return sentences
    
    # sentence를 받아 nouns를 return
    def get_nouns(self, sentences):
        nouns = []
        for sentence in sentences:
            if sentence is not '':
                nouns.append(' '.join([noun for noun in self.twitter.nouns(str(sentence))
            if noun not in self.stopwords and len(noun) > 1]))
        return nouns



# In[48]:


class GraphMatrix(object):
    def __init__(self):
        self.tfidf = TfidfVectorizer()
        self.cnt_vec = CountVectorizer()
        self.graph_sentence = []
    # 명사로 이루어진 문장 입력받아 setence graph를 return 
    def build_sent_graph(self, sentence):
        tfidf_mat = self.tfidf.fit_transform(sentence).toarray()
        self.graph_sentence = np.dot(tfidf_mat, tfidf_mat.T)
        return self.graph_sentence
    # 명사로 이루어진 문장 입력받아 word graph와 {idx:word} 형태의 딕셔너리를 return 
    def build_words_graph(self, sentence):
        cnt_vec_mat = normalize(self.cnt_vec.fit_transform(sentence).toarray().astype(float), axis=0)
        vocab = self.cnt_vec.vocabulary_
        return np.dot(cnt_vec_mat.T, cnt_vec_mat), {vocab[word] : word for word in vocab}


# In[49]:


class Rank(object):
    def get_ranks(self, graph, d=0.85): # d = damping factor
        A = graph
        matrix_size = A.shape[0]
        for id in range(matrix_size):
            A[id, id] = 0 # diagonal 부분을 0으로
            link_sum = np.sum(A[:,id]) # A[:, id] = A[:][id]
            if link_sum != 0:
                A[:, id] /= link_sum
            A[:, id] *= -d
            A[id, id] = 1

        B = (1-d) * np.ones((matrix_size, 1))
        ranks = np.linalg.solve(A, B) # 연립방정식 Ax = b
        return {idx: r[0] for idx, r in enumerate(ranks)}



# In[58]:


class TextRank(object):
    def __init__(self, text):
        self.sent_tokenize = SentenceTokenizer()
        
        if text[:5] in ('http:', 'https'):
            self.sentences = self.sent_tokenize.url2sentences(text)
        else:
            self.sentences = self.sent_tokenize.text2sentences(text)
        
        self.nouns = self.sent_tokenize.get_nouns(self.sentences)
        
        self.graph_matrix = GraphMatrix()
        self.sent_graph = self.graph_matrix.build_sent_graph(self.nouns)
        self.words_graph, self.idx2word = self.graph_matrix.build_words_graph(self.nouns)
        
        self.rank = Rank()
        self.sent_rank_idx = self.rank.get_ranks(self.sent_graph)
        self.sorted_sent_rank_idx = sorted(self.sent_rank_idx, key=lambda k: self.sent_rank_idx[k], reverse=True)
        
        self.word_rank_idx = self.rank.get_ranks(self.words_graph)
        self.sorted_word_rank_idx = sorted(self.word_rank_idx, key=lambda k: self.word_rank_idx[k], reverse=True)
        
    def summarize(self, sent_num=3):
        summary = []
        index=[]
            
        for idx in self.sorted_sent_rank_idx[:sent_num]:
            index.append(idx)
            
        index.sort()
        for idx in index:
            summary.append(self.sentences[idx])

        return summary

    def keywords(self, word_num=10):
        rank = Rank()
        rank_idx = rank.get_ranks(self.words_graph)
        sorted_rank_idx = sorted(rank_idx, key=lambda k: rank_idx[k], reverse=True)

        keywords = []
        index=[]
        for idx in sorted_rank_idx[:word_num]:
            index.append(idx)

        #index.sort()
        for idx in index:
            keywords.append(self.idx2word[idx])
            
        return keywords


# In[61]:


# url을 통해 문장 요약 

url = 'https://news.naver.com/main/read.nhn?mode=LSD&mid=shm&sid1=103&oid=437&aid=0000209082'
textrank = TextRank(url)
for row in textrank.summarize(3):
    print(row)
    print()
print('keywords :',textrank.keywords())

# text 파일을 통해 문장 요약 

file=open('a.txt','r', encoding='utf-8')
text=file.read()
file.close()

textrank = TextRank(text)
for row in textrank.summarize(3):
    print(row)
    print()
print('keywords :',textrank.keywords())


# In[ ]:





# In[ ]:




