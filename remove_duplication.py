#%%
import pandas as pd 
import itertools 
#%%
data=pd.read_pickle("first.pkl")

#%%
data.sort_values(by='title',inplace=True, ascending=False)

#%% 
data["title"].head(10)

#%%
indices=[]
for o in data.iterrows():
    indices.append(o)
#%%
indices=data.index.values.tolist()
print(indices[0])
#%%

not_similair=[]
i=0
j=0
number_of_row=data.shape[0]
while i < number_of_row and j < number_of_row:
    old_i=i
    a=data["title"].loc[indices[i]].split()
    i=j+1
    while j < number_of_row:
        b=data["title"].loc[indices[j]].split()
        lenght=max(len(a),len(b))
        count=0
        for k in itertools.zip_longest(a,b):
            if k[0]==k[1]:
                count=count+1
        if (lenght-count)>2: 
            not_similair.append(data["asin"].loc[indices[i]])
            if j== number_of_row-1:
                not_similair.append(data["asin"].loc[indices[j]])
            i=j
            break
        else:
            j=j+1
    if old_i==i: break
#%%
print(len(not_similair))

data=data.loc[data['asin'].isin(not_similair)]
print(data.shape)

#%%
indices=data.index.values.tolist()

not_similair=[]
number_of_row=data.shape[0]
while len(indices)!=0:
    i=indices.pop()
    print(i)
    not_similair.append(data["asin"].loc[i])
    a=data["title"].loc[i].split()
    for j in indices :
        b=data["title"].loc[j].split()
        lenght=max(len(a),len(b))
        count=0
    for k in itertools.zip_longest(a,b):
            if k[0]==k[1]:
                count=count+1
    if (lenght-count)>3:
        indices.remove(j)
print(len(not_similair))

data=data.loc[data['asin'].isin(not_similair)]
print(data.shape)
data.to_pickle("second.pkl")
#%%


