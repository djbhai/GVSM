# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 19:42:56 2019

@author: User
"""

import pickle
import numpy as np
import nltk.corpus
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Activation,Bidirectional,LSTM,Concatenate
from tensorflow.keras.layers import concatenate,Dropout,Masking
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
import time
import tensorflow as tf
import re
"""
samp = open("sample.txt",'wb')
dary = {}
zero = np.zeros([10,10])
wn = nltk.corpus.wordnet
sample=wn.lemma_from_key('accuracy%1:07:03::')  
sample=sample.synset()
sample = str(sample)
dary[sample]=zero
pickle.dump(dary,samp)
samp=open("sample.txt","rb")
da=pickle.load(samp)
print(da)
"""
def testStruct(struct):
    i=0
    for x in struct:
        if(i<1):
            print(x)
            i=i+1
        else:
            break
wn = nltk.corpus.wordnet
print(time.ctime())
wv = open("wordVocabEmb.txt","rb")
sv = open("senseVocabEmb.txt","rb")
wl = pickle.load(wv)
sl = pickle.load(sv)
print(time.ctime())
"""
if 'money' in wl:
    print(wl['money'])
"""
#print(len(sl))
senseVocab =[]
for x in sl:
    senseVocab.append(x)

senses = []
for x in sl:
    senses.append(sl[x])
sr = np.asarray(senses)
sr = sr.transpose()
print(sr.shape)
biases = np.zeros((26359,))
#biases = biases.transpose()

#print(len(sr))
"""
left_model = Sequential()
left_model.add(LSTM(1024, return_sequences=True, input_shape=(None,300)))
right_model = Sequential()
right_model.add(LSTM(1024,return_sequences=True,input_shape=(None,300)))

model = Sequential()
model.add(Concatenate([left_model,right_model]))
model.add(Dense(300,activation='elu'))
model.add(Dense(26048,activation='softmax',weights=sr))
"""
#prepare datasets
tagged_sentences = nltk.corpus.semcor.tagged_sents(tag='sem')
#testStruct(tagged_sentences)
pattern ='\'[a-zA-Z0-9\'-_.]*\''
leftData=[]
rightData=[]
output=[]
"""
for item in tagged_sentences:
    j=0
    for tree in item:
        if(isinstance(tree,nltk.tree.Tree)):
            if 'Lemma' in str(tree):
                lem=re.search(pattern,str(tree)).group(0)
                lem = lem[1:-1]
                #lem= lem.replace("'","")
                synset = wn.lemma(lem).synset()
                outputIndex=-1
                print(synset)
                print(senseVocab[0])
            #if synset in senseVocab:
                outputIndex=senseVocab.index(str(synset))
                leftWords=[]
                rightWords=[]
                for idx in range(j):
                    if(isinstance(item[idx],nltk.tree.Tree)):
                #print(tree.leaves())
                        wlist = tree.leaves()
                        if(len(wlist)==1):
                            if(wlist[0] in wl):
                                leftWords.append(wl[wlist[0]])
                        else:
                            wordPrep=wlist[0]
                            for word in wlist[1:]:
                                wordPrep+= "_"+word
                            if(wordPrep in wl):
                                leftWords.append(wl[wordPrep])
                    else:
                        if(item[idx][0] in wl):
                            leftWords.append(wl[item[idx][0]])
                for idx in range(j+1,len(item)):
                    if(isinstance(item[idx],nltk.tree.Tree)):
                #print(tree.leaves())
                        wlist = tree.leaves()
                        if(len(wlist)==1):
                            if(wlist[0] in wl):
                                rightWords.append(wl[wlist[0]])
                        else:
                            wordPrep=wlist[0]
                            for word in wlist[1:]:
                                wordPrep+= "_"+word
                            if(wordPrep in wl):
                                rightWords.append(wl[wordPrep])
                    else:
                        if(item[idx][0] in wl):
                            rightWords.append(wl[item[idx][0]])
                
                if(outputIndex>=0):
                    leftData.append(leftWords)
                    rightData.append(rightWords)
                    output.append(outputIndex)
        
        j=j+1 
print(output[0])
print(senseVocab[output[0]])
with open("left.txt", "wb") as fp:   #Pickling
    pickle.dump(leftData,fp)
with open("right.txt", "wb") as tp:   #Pickling
    pickle.dump(rightData,tp)
with open("output.txt", "wb") as rp:   #Pickling
    pickle.dump(output,rp)
"""

pseudoArray= []

for i in range(300):
    pseudoArray.append(0.8957583)

sample = np.asarray(pseudoArray)
print(np.shape(sample))

with open("senses.txt","rb") as oms:
    senseDat = pickle.load(oms)
with open("contexts.txt","rb") as omc:
    conDat = pickle.load(omc)

omsOutput = []
omsLeft =[]
omsRight = []
for s in senseDat:
    if(senseVocab.index(s)>=0):
        omsOutput.append(s)

for c in conDat:
    dat=c.split('#%')
    omsLeft.append(dat[0])
    omsRight.append(dat[1])

print(len(omsLeft))
print(len(omsRight))
print(len(omsOutput))



omsEmbLeft = []
omsEmbRight = []

for data in omsLeft:
    emb=[]
    wds = data.split(" ")
    if(len(wds)==1):
        if(wds[0]==''):
            emb.append(pseudoArray)
            omsEmbLeft.append(emb)
            continue
    for wd in wds:
        if(wd in wl):
            emb.append(wl[wd])
    omsEmbLeft.append(emb)
        
for data in omsRight:
    emb=[]
    wds = data.split(" ")
    if(len(wds)==1):
        if(wds[0]==''):
            emb.append(pseudoArray)
            omsEmbRight.append(emb)
            continue
    for wd in wds:
        if(wd in wl):
            emb.append(wl[wd])
    omsEmbRight.append(emb)
 
with open("omsLeft.txt",'wb') as olp:
    pickle.dump(omsEmbLeft,olp)
with open("omsRight.txt",'wb') as orp:
    pickle.dump(omsEmbRight,orp)
with open("omsOutput.txt","wb") as oop:
    pickle.dump(omsOutput,oop)

"""
with open("left.txt","rb") as lrp:
    op=pickle.load(lrp)
with open("right.txt","rb") as rrp:
    rp=pickle.load(rrp)
with open("output.txt","rb") as orp:
    oo = pickle.load(orp)


print(len(oo))
print(len(op))
print(len(rp))

#oo = np.asarray(oo)
print(time.ctime())
for x in rp:
    for item in x:
        for val in item:
            if val == 0.8957583:
                print('yes')
print(time.ctime())
lt = np.array(op[0])
rt= np.array(rp[0])
lt = [lt]
lt = np.array(lt)
rt = [rt]
rt= np.array(rt)
opt = np.array([oo[0]])
print(rt.shape)
print(lt.shape)
print(opt.shape)
print(type(lt).__name__)
print(type(rt).__name__)
left_tensor=keras.Input(shape=(None,300))
print(left_tensor.shape)
ldp = Dropout(rate = 0.3)(left_tensor)
lrn = LSTM(1024, input_shape=(None,300))(ldp)

right_tensor = keras.Input(shape=(None,300))
mask = Masking(mask_value=0.8957583,input_shape=(None,300))(right_tensor)
rdp = Dropout(rate=0.3)(mask)
rrn =LSTM(1024, input_shape=(None,300))(rdp)
merge = concatenate([lrn,rrn])

fully_connected = Dense(300,activation='elu')(merge)
softScores = Dense(26359,activation='softmax',weights=[sr,biases])(fully_connected)
#print(softScores.get_weights)
model = Model(inputs=[left_tensor,right_tensor],outputs = softScores)

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam')
for x in model.layers[6].get_weights():
    x = np.asarray(x)
    print(x.shape)
print(model.summary())
sample=[]
for i in range(0,300):
    sample.append(0.8957583)
sample = [[sample]]
sample = np.asarray(sample)
print(sample.shape)
model.fit([lt,rt],opt)
print(rt)
"""
"""
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam')
model.fit([lt,rt],oo[0])

#model.add(Bidirectional(LSTM(300)))

#model = Sequential()
"""