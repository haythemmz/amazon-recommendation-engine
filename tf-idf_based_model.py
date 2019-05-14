#%%
import math 
import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import pairwise_distances
import pickle
from numpy import linalg as LA
import operator
#%%
df=pd.read_pickle("third.pkl")

#%%
vectorizer = TfidfVectorizer(min_df=0)
title_tfidf=vectorizer.fit_transform(df['title'])
print(title_tfidf.get_shape())
print(type(title_tfidf))
title_tfidf=title_tfidf.toarray()
print(title_tfidf.shape)
#%%
def tfidf_based_model(q_id,num_near):
        title_tfidf[q_id].reshape(-1,1)
        pair_distances=pairwise_distances(title_tfidf,title_tfidf[q_id], n_jobs=-1)
        indices=np.argsort(pair_distances.flatten())[:num_near]
        pdists=np.sort(pair_distances.flatten())[:num_near]
        df_indices = list(df.index[indices])
        for i in range(0,len(indices)):
                print('ASIN :',df['asin'].loc[df_indices[i]])
                print ('Brand:', df['brand'].loc[df_indices[i]])
                print ('Title:', df['title'].loc[df_indices[i]])
                print ('Euclidean similarity with the query image :', pdists[i])
                print('='*60)
        return df_indices
#%%

df_tfidf=pd.DataFrame(data=title_tfidf,columns=list(vectorizer.get_feature_names()),index=df['asin'])
#%%
products=df_tfidf.index.tolist()
print(products[5])
print((df_tfidf.loc[['B004GSI2OS'],:]).values-(df_tfidf.loc[[products[5]],:]).values)
import time
start = time. time()
print(LA.norm( (df_tfidf.loc[['B004GSI2OS'],:]).values - (df_tfidf.loc[[products[5]],:]).values))
end = time. time()
print(end - start)

#%%
def similarity_function(product_id,df,number):
        similarity_dict={}
        i=0
        products=df.index.tolist()
        #print(products)
        print(product_id in products)
        while(i<number+1):
                if product_id !=products[i]:
                        a=LA.norm((df.loc[[product_id],:]).values-(df.loc[[products[i]],:]).values)
                        print(i)
                        similarity_dict[products[i]]=a
                i=i+1
        print(similarity_dict)
        m=max(list(similarity_dict.values()))
        max_key=max(similarity_dict.items(), key=operator.itemgetter(1))[0]
        #print("m={}, max={}".format(m,max_key))
        for j in products:
                if (j not in similarity_dict.keys()) and (product_id !=j):
                        b=LA.norm( (df.loc[[product_id],:]).values-(df.loc[[j],:]).values)
                        print("b={}".format(b))
                        if b < m :
                                del similarity_dict[max_key]
                               similarity_dict[j]=b
                                m=max(list(similarity_dict.values()))
                                max_key=max(similarity_dict.items(), key=operator.itemgetter(1))[0]
                                #print("m={}, max={}".format(m,max_key))


        return similarity_dict

#%%
start=time.time()
dict_res=similarity_function(product_id='B004GSI2OS',df=df_tfidf,number=3)
print(dict_res)
end=time.time()
print(end-start)
#%%
print(dict_res)


##########################################################################
#                                                                        #       
#                          IDF model                                     #       
#                                                                        #       
##########################################################################
#%%
wordFreqDic = {
    "Hello": 56,
    "at" : 23 ,
    "test" : 43,
    "this" : 43
    }
start = time. time()
m=max(list(wordFreqDic.values()))
max_key=max(wordFreqDic.items(), key=operator.itemgetter(1))[0]
print(m)
print(max_key)
end=time.time()
print(end-start)



#%%
def calculate_idf(word):
    a= sum(1 for j in df['title'] if word in j.split())
    
    idf=math.log(df.shape[0]/a)
    return idf

#%%
vocab=[]
for j in df['title']:
        vocab=set(vocab).union(set(j.split()))
#%%
idf_dict={}
for i in vocab :
        idf_dict[i]=calculate_idf(i)
#%%

print(idf_dict)
#%%
vocab_lenght=len(vocab)
sentences=df.shape[0]
#%%
idf_array=np.zeros((sentences,vocab_lenght))
#%%
idf_array.shape
#%%

df_idf=pd.DataFrame(data=idf_array,columns=vocab,index=df['asin'])
#%%
for j in df_idf.index : 
        a=df.loc[j,'title'].split()
        for i in a :
                df_idf.loc[j,i]=idf_dict[i]

#%%
df_idf.cov()


#%%
df_idf.to_pickle("idf_dataframe.pkl")

#%%
pickle_out = open("idf_dict.pkl","wb")
pickle.dump(idf_dict, pickle_out)
pickle_out.close()
#idf_dict.to_pickle("idf_dict.pkl")