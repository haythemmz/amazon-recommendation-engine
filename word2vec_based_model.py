
#%%
from gensim.models import Word2Vec
import math 
import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import pairwise_distances
import pickle

#%%
df = pd.read_pickle("third_without_stemming.pkl")

#%%
vocab=[]
for j in df['title']:
        vocab=set(vocab).union(set(j.split()))

#%%
def calculate_idf(word):
    a= sum(1 for j in df['title'] if word in j.split())
    
    idf=math.log(df.shape[0]/a)
    return idf
idf_without_stemming_dict={}
for i in vocab :
        idf_without_stemming_dict[i]=calculate_idf(i)
#%%
pickle_out = open("idf_without_stemming_dict.pkl","wb")
pickle.dump(idf_without_stemming_dict, pickle_out)
pickle_out.close()
#%%
idf_without_stemming_dict=pickle.load( open( "idf_without_stemming_dict.pkl", "rb" ) )
print(idf_without_stemming_dict)


#%%
sent=list(df['title'])
sentces=[]
for j in sent:
    sentces.append(j.split())

#%%
sentces[:5]
#%%
model2 = Word2Vec(sentences=sentces, min_count = 1, size = 50, window = 5, sg = 1) 
#%%
#w2v_model.train(sentences,epochs=10,total_examples=len(sentences))
#%%
words = list(model2.wv.vocab)
print(words)

#%%
word2vec_dict={}
for j in words:
        word2vec_dict[j]=model2.wv.get_vector(j)
#%%
print(len(words))
print(word2vec_dict[list(word2vec_dict.keys())[0]].shape)
#%%
pickle_out = open("word2vect_dict.pkl","wb")
pickle.dump(word2vec_dict, pickle_out)
pickle_out.close()

#%%
word2vec_dict=pickle.load( open( "word2vect_dict.pkl", "rb" ) )
#%%
#vocab_lenght=len(words)
sentences=df.shape[0]
x=word2vec_dict[list(word2vec_dict.keys())[0]].shape[0]
#y=word2vec_dict[list(word2vec_dict.keys())[0]].shape[1]
word2vect_array=np.zeros((sentences,x))
print(word2vect_array.shape)

#%%

l=[]
for j in range(x):
        l.append('w'+str(j))
print(l)
#%%
print(len(l))
#%%
df_w2v_mean=pd.DataFrame(data=word2vect_array,columns=l,index=df['asin'])

#%%
#def construct_w2v_matrix(df_w2v_mean,df):
a=df.set_index('asin')
print(len(list(a.index)))
#%%
## need to convert to function 
title_dict=dict(zip(list(df['asin']), list(df['title'])))
x=df_w2v_mean.shape[1]
for j in df_w2v_mean.index:
        sen=title_dict[j]
        l=sen.split()
        arr=np.zeros((1,x))
        for k in l :
                if k in word2vec_dict.keys():
                        arr=arr+word2vec_dict[k].reshape((1,x))
        m=arr*(1/len(l))
        df_w2v_mean.loc[j,:]=m     
                
#%%
df_w2v_mean.to_pickle("word2vect_average.pkl")
#%%
df_w2v_idf=pd.DataFrame(data=word2vect_array,columns=l,index=df['asin'])
#%%
x=df_w2v_idf.shape[1]
for j in df_w2v_mean.index:
        sen=title_dict[j]
        l=sen.split()
        arr=np.zeros((1,x))
        for k in l :
                if k in word2vec_dict.keys():
                        arr=arr+idf_without_stemming_dict[k]*word2vec_dict[k].reshape((1,x))
        
        df_w2v_idf.loc[j,:]=arr  
#%%
df_w2v_idf.to_pickle("word2vect_idf.pkl")