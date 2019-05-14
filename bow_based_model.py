#%%
import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances
#%%
df=pd.read_pickle("third.pkl")
#%%

vector=CountVectorizer()
title_bag=vector.fit_transform(df['title'])
print(title_bag.get_shape())
title_bag=title_bag.toarray()
print(title_bag.shape)

#%%
def bag_based_model(q_id,num_near):
        title_bag[q_id].reshape(-1,1)
        pair_distances=pairwise_distances(title_bag,title_bag[q_id], n_jobs=-1)
        indices=np.argsort(pair_distances.flatten())[:num_near]
        pdists=np.sort(pair_distances.flatten())[:num_near]
         df_indices = list(df.index[indices])
        for i in range(0,len(indices)):
                print('ASIN :',df['asin'].loc[df_indices[i]])
                print ('Brand:', df['brand'].loc[df_indices[i]])
                print ('Title:', df['title'].loc[df_indices[i]])
                print ('Euclidean similarity with the query image :', pdists[i])
                print('='*60)

#%%
bag_based_model(931,5)

#%%
t=title_bag.toarray()
#%%
t[931].reshape(-1,1)
#%%
pair_distances=pairwise_distances(t,t[931].reshape(-1,1), n_jobs=-1)