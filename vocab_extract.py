# -*- coding: utf-8 -*-
"""
Created on Wed May 29 21:46:09 2019

@author: User
"""
from os import listdir
from xml.dom import minidom
import nltk.corpus
import re
import string
import tarfile
import xml.dom
import gensim
from gensim.test.utils import get_tmpfile
import numpy as np
import time 
import pickle

print(time.ctime())
model = gensim.models.KeyedVectors.load_word2vec_format('C:\\Users\\User\\WPy64-3720\\notebooks\\GoogleNews-vectors-negative300.bin', binary=True)  
embeddings_index = {}
wordVocDict={}
senseVocDict={}
for word, vector in zip(model.vocab, model.vectors):
    coefs = np.asarray(vector, dtype='float32')
    embeddings_index[word] = coefs
    
tagged_sentences= nltk.corpus.semcor.tagged_sents(tag='sem')
wn = nltk.corpus.wordnet
i=0
pattern ='\'[a-zA-Z0-9\'-_.]*\''
wordVocab = []
senseVocab=[]
senseVocab=set(senseVocab)
wordVocab=set(wordVocab)
i=0
"""
for item in tagged_sentences:
    if(i>20):
        break
    print(item)
    i=i+1
"""
#generate sense vocab
i=0

#print(tagged_sentences[0])
"""
for item in tagged_sentences:
    for tree in item:
        if 'NE' in str(tree):
            print(str(tree))
"""

for item in tagged_sentences:
    for tree in item:
        #if 'NE' not in str(tree):
        if(isinstance(tree,nltk.tree.Tree)):
            #print(tree.leaves())
            wlist = tree.leaves()
            if(len(wlist)==1):
                wordVocab.add(wlist[0])
            else:
                wordPrep=wlist[0]
                for word in wlist[1:]:
                    wordPrep+= "_"+word
                wordVocab.add(wordPrep)
        else:
            wordVocab.add(tree[0])

for item in tagged_sentences:
    for tree in item:
        if 'Lemma' in str(tree):
            lem=re.search(pattern,str(tree)).group(0)
            lem = lem[1:-1]
            #lem= lem.replace("'","")
            synset = wn.lemma(lem).synset()
            senseVocab.add(synset)

    i=i+1

print("The Semcor Sense Vocabulary:")
print(len(senseVocab))

path = "omsti//one-million-sense-tagged-instances-wn30.tar//one-million-sense-tagged-instances-wn30//noun"
nounList = listdir("omsti//one-million-sense-tagged-instances-wn30.tar//one-million-sense-tagged-instances-wn30//noun")

#print(nounList)
i=0
j=0
wn = nltk.corpus.wordnet
senses =[]
scs=[]
allSenses=[]
sample=wn.lemma_from_key('accuracy%1:07:03::')  
sample=sample.synset()
#sample=wn.synset_from_sense_key('accuracy%1:07:02::')  
#print(sample)  
print(sample)  
path = "omsti//one-million-sense-tagged-instances-wn30.tar//one-million-sense-tagged-instances-wn30//noun"
nounList = listdir("omsti//one-million-sense-tagged-instances-wn30.tar//one-million-sense-tagged-instances-wn30//noun")

verbPath = "omsti//one-million-sense-tagged-instances-wn30.tar//one-million-sense-tagged-instances-wn30//verb"
verbList = listdir("omsti//one-million-sense-tagged-instances-wn30.tar//one-million-sense-tagged-instances-wn30//verb")

advPath = "omsti//one-million-sense-tagged-instances-wn30.tar//one-million-sense-tagged-instances-wn30//adv"
advList = listdir("omsti//one-million-sense-tagged-instances-wn30.tar//one-million-sense-tagged-instances-wn30//adv")

adjPath = "omsti//one-million-sense-tagged-instances-wn30.tar//one-million-sense-tagged-instances-wn30//adj"
adjList = listdir("omsti//one-million-sense-tagged-instances-wn30.tar//one-million-sense-tagged-instances-wn30//adj")


def dataPrepare(fileList,dirPath):
    i=0
    for item in fileList:
        if(i%2 ==0):
            i=i+1
            continue        
        else:
            wordXml=minidom.parse(dirPath+"//"+item)
            instances=wordXml.getElementsByTagName("instance")
            contexts =wordXml.getElementsByTagName("context")
            iContexts =[]
            for context in contexts:
                c = context.childNodes
                text=""
                for nds in c:
                    if(isinstance(nds,xml.dom.minidom.Element)):
                        #add a seperator where the sense tagged instance appears
                            text+= "#%"
                            continue
                    else:
                        text+=nds.data
                iContexts.append(text)          #a files contexts being appended
                #print(iContexts)
            scs.append(iContexts)
            iteration=[]
            for sense in instances:
                iteration.append(sense.getAttribute("id"))
            senses.append(iteration)
        #print(iteration)
            i=i+1
    i=0
    for item in fileList:
        if(i%2==0):
            keyFile= open(dirPath+"//"+item,'r')
            fileSense = []
            for key in keyFile:
                line = key.split(" ")
                oneSense = line[2]
                oneSense=oneSense.replace("\n","")
                lemmar =wn.lemma_from_key(oneSense)
                oneSense=lemmar.synset()
                fileSense.append(oneSense)              #file senses being appended
            allSenses.append(fileSense)
            i=i+1
        else:
            i=i+1
            continue

dataPrepare(adjList,adjPath)
dataPrepare(advList,advPath)
dataPrepare(nounList,path)
dataPrepare(verbList,verbPath)
#sample=wn.synset_from_sense_key(allSenses[0][0])  
#print(sample)           
print(allSenses[0])
print(len(senses))
print(scs[0][0])
for item in allSenses:
    for x in item:
        senseVocab.add(x)
for cs in scs:
    for c in cs:
        wds = c.split(" ")
        for wd in wds:
            wordVocab.add(wd)
print("#%" in wordVocab)
print(senses[0])
print(len(wordVocab))
print(len(senseVocab))
print(len(model.vocab))
wvf = open("wordVocabEmb.txt","wb")
svf = open("senseVocabEmb.txt","wb")
zs=0
for word in wordVocab:
    if(word in embeddings_index):
        wordVocDict[word]=embeddings_index[word]
for sense in senseVocab:
    definition = sense.definition()
    comps = definition.split(" ")
    acs = 0
    avg=np.zeros(coefs.shape)
    for comp in comps:
        if(comp in embeddings_index):
            acs=acs+1
            avg= avg+embeddings_index[comp]
    if(acs ==0):
        zs=zs+1
        avg = np.zeros(coefs.shape)
    else:
        avg =np.divide(avg,acs)
    sense = str(sense)
    senseVocDict[sense]=avg
pickle.dump(senseVocDict,svf)            
pickle.dump(wordVocDict,wvf)
print(time.ctime())
print("zero senses: "+str(zs))
print("len of wv dict:"+str(len(wordVocDict)))
print("len of sense voc dict: "+str(len(senseVocDict)))
#print(len(tagged_sentences))

"""
senseFile = open("C:\\Users\\User\\WPy64-3720\\notebooks\\Sense_Annotated"
              +"\\WSD_Training_Corpora\\SemCor+OMSTI\\semcor+omsti.data.xml","r")
i=0
for unit in senseFile:
    if(i>=20):
        break
    print(unit)
    i=i+1


corpuString=""
for unit in senseFile:
    corpuString+=unit

print("part2")    
stringPar = minidom.parseString(corpuString)

sensePar = minidom.parse("C:\\Users\\User\\WPy64-3720\\notebooks\\Sense_Annotated"
                 +"\\WSD_Training_Corpora\\SemCor\\semcor.data.xml")
numberOfTexts= sensePar.getElementsByTagName('text')
sents =sensePar.getElementsByTagName('sentence')
wfs = sensePar.getElementsByTagName('wf')
instances = sensePar.getElementsByTagName('instance')
print(len(numberOfTexts)+len(sents)+len(wfs)+len(instances))
"""
