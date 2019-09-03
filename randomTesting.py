# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 21:19:15 2019

@author: User
"""
import wordnetUtils
import nltk.corpus
import numpy as np
from scipy.sparse import csr_matrix
from os import listdir
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus.reader.wordnet import Lemma,Synset 
from nltk.corpus import stopwords
import re
import pickle
from spellchecker import SpellChecker
import signal

k=0
p=0
wnUtils = wordnetUtils.wordnetUtils(k,p)
wn= nltk.corpus.wordnet

InputFileNames = listdir("MoonLander/intro-dataset/high")
OutputFileNames = listdir("MoonLander/intro-dataset/low")

srL=[]

def kbih(signal,frame):
    with open("srList.txt","wb") as fp:
        pickle.dump(srL,fp)
    exit(0)
signal.signal(signal.SIGINT, kbih)

def get_wordnet_pos(listOfTokens):
    """Map POS tag to first character lemmatize() accepts"""
    tag_dict = {"J": wn.ADJ,
                "N": wn.NOUN,
                "V": wn.VERB,
                "R": wn.ADV}
    posTokens= nltk.pos_tag(listOfTokens)
    wposTokens = []
    for token in posTokens:
        tag = token[1][0][0].upper()
        tag= tag_dict.get(tag,wn.NOUN)
        wposTokens.append((token[0],tag))  
    return wposTokens

Vocab =set([])
Untagged = set([])

def dataPreProcessing(inputs,outputs,includeUT):
    pathToInputFiles = []
    pathToOutputFiles  = []
    inputDict = {}
    outputDict ={}
    totalTokens =0
    with open("senseMappings.txt","rb") as fp:
        senseMappings = pickle.load(fp)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    for file in inputs:
        pathToInputFiles.append("MoonLander/intro-dataset/high/"+file)
    for file in outputs:
        pathToOutputFiles.append("MoonLander/intro-dataset/low/"+file)
    for index,file in zip(inputs,pathToInputFiles):
        fp = open(file,'r')
        content = fp.read()
        tokens = nltk.word_tokenize(content)
        wposTokens=[(lemmatizer.lemmatize(w[0],w[1]),w[1]) for w in get_wordnet_pos(tokens)]
        woSwords = []
        for token in wposTokens:
            if(token[0] not in stop_words):
                if(re.search('\A^[a-zA-Z0-9]',token[0])):
                    #totalTokens+=1
                    if(token[0].lower()+'.'+token[1] in senseMappings):
                        Vocab.add(token[0].lower()+'.'+token[1])
                        totalTokens+=1
                        woSwords.append(senseMappings[token[0].lower()+'.'+token[1]])
                    elif(includeUT):
                        Untagged.add(token[0].lower()+'.'+token[1])
                        woSwords.append(token[0].lower()+'.'+token[1])
                    inputDict[index]=woSwords
    for index,file in zip(outputs,pathToOutputFiles):
        fp = open(file,'r')
        content = fp.read()
        tokens = nltk.word_tokenize(content)
        wposTokens=[(lemmatizer.lemmatize(w[0],w[1]),w[1]) for w in get_wordnet_pos(tokens)]
        woSwords = []
        for token in wposTokens:
            if(token[0] not in stop_words):
                if(re.search('\A^[a-zA-Z0-9]',token[0])):
                    if(token[0].lower()+'.'+token[1] in senseMappings):
                        Vocab.add(token[0].lower()+'.'+token[1])
                        totalTokens+=1
                        woSwords.append(senseMappings[token[0].lower()+'.'+token[1]])
                    elif(includeUT):
                        woSwords.append(token[0].lower()+'.'+token[1])

                    #totalTokens+=1
                    outputDict[index]=woSwords
    ppdata =[inputDict,outputDict]
    #print(senseMappings)
    return ppdata



def wordnetTagger(vocabulary):
    synsetDict = {}
    for word in vocabulary: 
        wpos =word.split('.')
        wd =wpos[0]
        pos = wpos[1]
        synsetPos = []
        synset =[]
        if(wn.synsets(wd)):
            print(wd)
            synset = wn.synsets(wd)
        
        for ss in synset:
            if(ss.pos()==pos):
                synsetPos.append(ss)
        i=1
        for ss in synsetPos:
            print(str(i)+" "+ss.definition())
            i=i+1
        print("enter your choice:")
        if(synsetPos):
            choice = int(input())
            tss = synsetPos[choice-1]
            nameofSynset = tss.name()
            synsetDict[word] = nameofSynset
    with open("senseMappings.txt","wb") as fp:
        pickle.dump(synsetDict,fp)
            

def Tokenizer(s):
        regexp = '(\w)*\.[avrns]\.[0-9]*'
        wds = s.split(" ")
        return wds
        
  
def computeSrs(features):
    n1=0
    itr = 0
    while(n1 < len(features)-1):
        n2 = n1+1
        while(n2 <=len(features)-1):
            print("Iteration: "+str(itr))
            srL.append(wnUtils.sr(wn.synset(features[n1]),wn.synset(features[n2])))
            itr+=1
            n2+=1
        n1+=1
    
    return srL
    
    
def getTfIdf(inputDict,outputDict):
    inputs =[]
    outputs=[]
    vocab=dict({})
    i=0
    for key in inputDict:
        inputString=""
        j=0
        for word in inputDict[key]:
                if(j+1 == len(inputDict[key])):
                    inputString+=word
                else:
                    inputString+=word+" "
                    j=j+1
                if(word not in vocab):
                    vocab[word]=i
                    i=i+1
                
        inputs.append(inputString)
    for key in outputDict:
        outputString=""
        j=0
        for word in outputDict[key]:
            if(j+1 == len(outputDict[key])):
                    outputString+=word
            else:
                outputString+=word+" "
                j=j+1
        outputs.append(outputString)
    #print(inputs)
    #print(outputs)
    
    """
    pathToInputFiles =[]
    for file in inputs:
        pathToInputFiles.append("MoonLander/intro-dataset/high/"+file)
    pathToOutputFiles =[]
    for file in outputs:
        pathToOutputFiles.append("MoonLander/intro-dataset/low/"+file)
    """
    vectorizer = TfidfVectorizer(input='content',norm=None,vocabulary=vocab,
                                 token_pattern='(\w)*\.[avnsr]\.[0-9]*',tokenizer=Tokenizer)
    documentMapping = vectorizer.fit_transform(inputs)
    queryMapping = []
    queryMapping = vectorizer.transform(outputs)
    features = vectorizer.get_feature_names()
    #print(idfs[0:14])
    #print(features[0:14])
    #print(vectorizer.vocabulary_)
    tfIdfs = []
    tfIdfs.append(documentMapping)
    tfIdfs.append(queryMapping)
    tfIdfs.append(features)
    return tfIdfs

def TfIdfTest():
    # algorithm n(n-1)/2 dimensions
    # includeUT is 3rd positional argument false
    ppdata = dataPreProcessing(InputFileNames,OutputFileNames,False)
    tfIdfs = getTfIdf(ppdata[0],ppdata[1])
    tfIdfsDense=[]
    for item in tfIdfs:
        if(isinstance(item,list)):
            break    # break when features are read
        tfIdfsDense.append(item.todense())
    itr = 0
    for item in tfIdfsDense:
        if(isinstance(item,list)):
            break    # break when features are read
        row = item
        for it in row:
            print(it)
            itr+=1
            if(itr>1):
                break
        break
    """
    for item in tfIdfs:
        print(csr_matrix(item).toarray())
        break
    """
# algorithm n(n-1)/2 dimensions
def cosineSim2():
    # includeUT is 3rd positional argument false for this alg
    ppdata = dataPreProcessing(InputFileNames,OutputFileNames,False)
    tfIdfs = getTfIdf(ppdata[0],ppdata[1])
    features = tfIdfs[2]
    tfIdfsDense=[]
    for item in tfIdfs:
        if(isinstance(item,list)):
            break    # break when features are read
        tfIdfsDense.append(item.todense())
    srs=computeSrs(features)
    print(len(srs))
    #should be 120*119/2 for n = 120, 7140
#wordnetGraph():
cosineSim2()
#regexp is actually redundant as tokens are seperated by space
def regexpTest():
    reg = '((\w)*\.[avnsr]\.[0-9]*)|((\w)*\.[avnsr])'
    #some other regexp must be written to ensure that it get's all the vocab
    op = re.match(reg,sample)
    print(op)
    



def TaggingLeftOvers():
    dataPreProcessing(InputFileNames,OutputFileNames)
    print(Untagged)
    print(len(Untagged))
    # just load sensemappings,add mappings to taggable words.
    # insert in sensemappings dictionary and dump.
    """
    pos = {wn.ADJ,wn.NOUN,wn.VERB,wn.ADV}
    spell=SpellChecker()
    leftOverDict =dict([])
    #vocabulary = Vocab
    #wordnetTagger(vocabulary)
    with open('senseMappings.txt','rb') as fp:
        l = pickle.load(fp)
    print(l)
    print(len(l))
    print(len(Vocab))
    leftOver = set([])
    for k in Vocab:
        if k not in l:
            leftOver.add(k)
    print(len(leftOver))
    print(leftOver)
    
    for word in leftOver:
        lem= word.split(".")
        wd = lem[0]
        wdpos = lem[1]
        synsets=wn.synsets(wd)
        if(len(synsets)==0):
            correction=spell.correction(wd)
            synsets = wn.synsets(correction)
            if(len(synsets)!=0):
                i=1
                print(correction)
                print("enter your choice:")
                for ss in synsets:
                    print(str(i)+ " "+ss.definition())
                    i=i+1
                print(str(0)+" "+"continue")
                choice = int(input())
                if(choice == 0):
                    continue
                tss =synsets[choice-1]
                name = tss.name()
                leftOverDict[word] =name
        else:
            i=1
            print(word)
            print("enter your choice:")
            for ss in synsets:
                print(str(i)+ " "+ss.definition())
                i=i+1
            print(str(0)+" "+"continue")
            choice = int(input())
            if(choice ==0):
                continue
            tss =synsets[choice-1]
            name = tss.name()
            leftOverDict[word] =name
    with open('leftOverMappings.txt','wb') as fp:
        pickle.dump(leftOverDict,fp)
      
    
    with open('leftOverMappings.txt','rb') as fp:
        lo = pickle.load(fp)
    for mapp in lo:
        l[mapp]=lo[mapp]
    with open('senseMappings.txt','wb') as fp:
        pickle.dump(l,fp)
    print(len(l))
    print(len(lo))
    with open('senseMappings.txt','rb') as fp:
        l=pickle.load(fp)
    print(l)
    """
#TaggingLeftOvers()
#wordnetTaggerTest()

def test():
    op = getTfIdf(InputFileNames,OutputFileNames)
    for o in op:
        print(csr_matrix(o).toarray())

def wordCount():
    lemma = wn.words()
    lemma =list(lemma)
    print(len(lemma))

# Global Variables
lemma = wn.words()
allSynsets =wn.all_synsets() # get all synsets
allSynsets = list(allSynsets)
wordToCandidates = {}


def prepareData():
    for word in lemma:
        synsets = wn.synsets(word)
        senses =[]
        for s in synsets:
            sense =[]
            pos = s.pos()
            for lem in s.lemmas():
                ls = lem.synset()
                ln = lem.name()
                lns = wn.synsets(ln)
                nlns=[]
                for ss in lns:
                    if(ss.pos()==pos):
                        nlns.append(ss)
                li  = nlns.index(ls)+1
                triple=(ln,li,pos)
                sense.append(triple)
            senses.append(sense)
        wordToCandidates[word]=senses
        
        

def prevTest():
    ds = wn.synsets('dog')
    # synset of last sense Synset('chase.v.01')
    print(ds[7].lemmas())
    # where does 'tracks synset,chase.v.01 ,occur in synsets for the lemma'   
    ts = wn.synsets('track')
    print(ts) # occurs at 13th index
    #removing noun senses from the ts
    nm = []
    for t in ts:
        if(t.pos()=='v'):
            nm.append(t)
        
    
    ss= wn.synset('chase.v.01')
    print(nm.index(ss)+1)
    print(nm)



#where does lemma's synset occur in the set of synsets for the lemma
#get lemma,synset from Lemma, get synsets of lemma, get where synset occurs in lemma.
