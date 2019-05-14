#%%
import pandas as pd 

import codecs
from collections import Counter

import os
print(os.listdir())
#%%
#data = pd.read_json(codecs.open('tops_fashion.json','r','utf-8'))
data = pd.read_json('tops_fashion.json')
#%%

def missing_data_function(frame):
        total = frame.isnull().sum().sort_values(ascending=False)
        percent = (frame.isnull().sum()*100 / frame.isnull().count()).sort_values(ascending=False)
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        return missing_data
# this function calculate the number and the percentage of missing values in each column in the dataset 

#%%
print(missing_data_function(data))

#%%
print(data.shape)
#print ('Number of data points : ', data.shape[0])
#print('Number of features/variables:', data.shape[1])
print(data.keys())


#%%
print(data.columns)

#%%
data = data[['asin', 'brand', 'color', 'medium_image_url', 'product_type_name', 'title', 'formatted_price']]


#%%
def features_describtion(df,feature_name):
        print("***"+feature_name+"*****")
        print(df[feature_name].describe())
        print("*********************")

for j in data.columns : 
        features_describtion(data,j)
#%%%
def most_common(df,feature_name,number):
        return(Counter(list(df[feature_name])).most_common(number))

print(most_common(data,'product_type_name',10))


#%%
def drop_rows(df,remove_list):
        df=df.dropna(subset=remove_list)
        return df

data=drop_rows(data,["formatted_price",'color'])
#%%
print(data.shape)

#%%
print(sum(data.duplicated('title')))

#%%
data["length_title"]=data["title"].apply(lambda x: len(x.split()) )

#%%
data.groupby("length_title")["length_title"].count().plot.bar()

#%%
data=data[data["length_title"]>4]
#%%
missing_data_function(data)
#%%
data.shape
data.to_pickle("first.pkl")