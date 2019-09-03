# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 02:30:59 2019

@author: User
"""
from os import listdir
import tarfile
import nltk.corpus
from xml.dom import minidom
import xml.dom
import pickle

path = "omsti//one-million-sense-tagged-instances-wn30.tar//one-million-sense-tagged-instances-wn30//noun"
nounList = listdir("omsti//one-million-sense-tagged-instances-wn30.tar//one-million-sense-tagged-instances-wn30//noun")

verbPath = "omsti//one-million-sense-tagged-instances-wn30.tar//one-million-sense-tagged-instances-wn30//verb"
verbList = listdir("omsti//one-million-sense-tagged-instances-wn30.tar//one-million-sense-tagged-instances-wn30//verb")

advPath = "omsti//one-million-sense-tagged-instances-wn30.tar//one-million-sense-tagged-instances-wn30//adv"
advList = listdir("omsti//one-million-sense-tagged-instances-wn30.tar//one-million-sense-tagged-instances-wn30//adv")

adjPath = "omsti//one-million-sense-tagged-instances-wn30.tar//one-million-sense-tagged-instances-wn30//adj"
adjList = listdir("omsti//one-million-sense-tagged-instances-wn30.tar//one-million-sense-tagged-instances-wn30//adj")



sam = "bab#%bad"
bb = sam.split("#%")
print(bb)
#print(nounList)
i=0
j=0
wn = nltk.corpus.wordnet
senses =[]
scs=[]
allSenses=[]

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
    


sample=wn.lemma_from_key('accuracy%1:07:03::')  
sample=sample.synset()
#sample=wn.synset_from_sense_key('accuracy%1:07:02::')  
#print(sample)  
print(sample) 
dataPrepare(adjList,adjPath)
dataPrepare(advList,advPath)
dataPrepare(nounList,path)
dataPrepare(verbList,verbPath)
#sample=wn.synset_from_sense_key(allSenses[0][0])  
#print(sample) 
output = []
data = []
for item in allSenses:
    for x in item:
        output.append(str(x))
for item in scs:
    for x in item:
        data.append(x)
print(allSenses[0])
print(len(senses))
print(len(scs[0]))
print(len(scs))
print(len(allSenses))
print(senses[0])
with open("senses.txt","wb") as sp:
    pickle.dump(output,sp)
with open("contexts.txt","wb") as cp:
    pickle.dump(data,cp)
print(len(data))
print(len(output))
#print(scs[0][1])
#print(scs[0][2])
"""
Omsti=tarfile.open("omsti//one-million-sense-tagged-instances-wn30.tar//one-million-sense-tagged-instances-wn30.tar")
omsti_content = Omsti.getmembers()
i=0
for dat in omsti_content:
    #print(dat)
    i=i+1
    if(dat.isfile()):
        print(dat.name)
        f=Omsti.extractfile(dat)
        content = f.read()
"""
"""
    f = Omsti.extractfile(dat)
    content = f.read()
    print(content)
"""
"""
    #if(i>10):
    #    break
print(i)   
"""