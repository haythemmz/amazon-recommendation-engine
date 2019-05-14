#%%
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

#%%
df=pd.read_pickle('third.pkl')
#%%

df.keys()
#%%
df['brand'].fillna(value="not_available", inplace=True)

brands= [ x.replace(" ","_") for x in df['brand'].values]
product_type_name= [ x.replace(" ","_") for x in df['product_type_name'].values]
color= [ x.replace(" ","_") for x in df['color'].values]

#%%
vector_br=CountVectorizer()
brands_vector=vector_br.fit_transform(brands)

#%%
vector_pr=CountVectorizer()
product_type_name_vector=vector_pr.fit_transform(product_type_name)

#%%
vector_co=CountVectorizer()
color_vector=vector_co.fit_transform(color)

#%%
def convert_to_df(v, v_c):
    v_array=v.toarray()
    df_v=pd.DataFrame(data=v_array,columns=v_c.get_feature_names(), index=df['asin'])
    return df_v
#%%
df_brand=convert_to_df(v=brands_vector,v_c=vector_br)

#%%
df_color=convert_to_df(v=color_vector,v_c=vector_co)