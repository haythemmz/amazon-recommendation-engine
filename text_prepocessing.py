#%%
import pandas as pd 
import nltk 
from nltk.corpus import stopwords
import re
#%%
df=pd.read_pickle("second.pkl")
#%%
df.keys()
#%%
nltk.download("stopwords")
Stopwords = set(stopwords.words('english'))
print(Stopwords)
#%%
abbr_dict={
        "isn't":"is not",
        "wasn't":"was not",
        "aren't":"are not",
        "weren't":"were not",
        "can't":"can not",
        "couldn't":"could not",
        "don't":"do not",
        "didn't":"did not",
        "shouldn't":"should not",
        "wouldn't":"would not",
        "doesn't":"does not",
        "haven't":"have not",
        "hasn't":"has not",
        "hadn't":"had not",
        "won't":"will not",
        '["\',\.<>()=*#^:;%µ?|&!-123456789]':""}


# In[ ]:


l=[]
for sentence in (df['title'].values):
    sentence = sentence.lower()                 
    cleanr = re.compile('<.*?>')
    sentence = re.sub(cleanr, ' ', sentence)        
    for j in abbr_dict.keys():
                sentence=re.sub(j,abbr_dict[j],sentence)
    l.append(sentence)

#%%
l[0]
#%%

def preprocessing(l,stemming):
    snow = nltk.stem.SnowballStemmer('english')
    a=[]
    for j in l :
        if stemming==True:
            a.append([snow.stem(word) for word in j.split() if word not in Stopwords])
        else:
            a.append([word for word in j.split() if word not in Stopwords])        
    return a  

#%%
a=preprocessing(l,stemming=True)
#%%
b=preprocessing(l,stemming=False)
#%%

def to_sentence(l):
    sentence = []
    for row in l:
        sequ = ''
        for word in row:
            sequ = sequ + ' ' + word
        sentence.append(" ".join(sequ.split()))
    return sentence
#%%
a=to_sentence(a)
df['title']=a
df.to_pickle("third.pkl")
#%%
b=to_sentence(b)
df['title']=b
df.to_pickle("third_without_stemming.pkl")





#%%
b[0]
#%%
print(df['title'].iloc[0])