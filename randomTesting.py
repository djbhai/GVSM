# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 21:19:15 2019

@author: User
"""
import wordnetUtils
import nltk.corpus
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine
from os import listdir
import os
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus.reader.wordnet import Lemma,Synset 
from nltk.corpus import stopwords
from tkinter import filedialog
import re
import pickle
from spellchecker import SpellChecker
import signal
import concurrent.futures


k=0
p=0
wnUtils = wordnetUtils.wordnetUtils(k,p)
wn= nltk.corpus.wordnet
If = filedialog.askdirectory(initialdir=os.getcwd(),title="select high directory")
print(If)
Of = filedialog.askdirectory(initialdir=os.getcwd(),title="select low directory")
print(Of)
InputFileNames= listdir(If)
OutputFileNames= listdir(Of)
srDict={}      #To store sr's in index numbered dict
Vocab =set([])
Untagged = set([])
inputDict={}
outputDict={}
traceLinks =[(1,0),(3,1),(4,2),(5,0),(7,3),(9,4),(6,3),(7,4)]  #moon lander


def logistics():
    maximum = 120
    n= 0
    sums = 0
    while(sums<=4760):
        prev= sums
        sums+= maximum-(n+1)
        n+=1
    print("N1 is "+str(n))
    max1 = maximum - n
    print(max1)
    
    print("sum is " + str(prev))
#logistics()   

def Tokenizer(s):
        regexp = '(\w)*\.[avrns]\.[0-9]*'
        wds = s.split(" ")
        return wds

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



def tokenize(inputs,outputs):
    pathToInputFiles = []
    pathToOutputFiles  = []
    tokenizedDict={}
    lemmatizer = WordNetLemmatizer()
    for file in inputs:
        pathToInputFiles.append(If+"/"+file)
    
    for index,file in zip(inputs,pathToInputFiles):
        fp = open(file,'r')
        content = fp.read()
        tokens = nltk.word_tokenize(content)
        wposTokens=[(lemmatizer.lemmatize(w[0],w[1]),w[1]) for w in get_wordnet_pos(tokens)]
        tokenizedDict[index]=wposTokens
    
    return tokenizedDict


def dataPreProcessing(inputs,outputs,includeUT):
    pathToInputFiles = []
    pathToOutputFiles  = []
    lemmatizer = WordNetLemmatizer()
    totalTokens =0
    senseMappingsFile = filedialog.askopenfilename(initialdir=os.getcwd(),
                                               title="Select Sense Mappings File")
    with open(senseMappingsFile,"rb") as fp:
        senseMappings = pickle.load(fp)
    stop_words = set(stopwords.words('english'))
        
    wposDict = tokenize(inputs,outputs)
    woSwords = []
    for index,sentence in wposDict.items():
        for token in sentence:
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
    for file in outputs:
        pathToOutputFiles.append(Of+"/"+file)
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
    print(ppdata)
    #print(senseMappings)
    return ppdata



def wordnetTagger():
    vocabulary = dict()
    entered = set()
    tokenizedDict  = tokenize(InputFileNames,OutputFileNames)
    for key in tokenizedDict:
        wposTokens=tokenizedDict[key]
        for token in wposTokens:
            entry=  token[0]+"."+token[1]
            if(len(vocabulary)!=0):
                entered=[item for item in vocabulary if item==entry ]
            if(len(entered)==0):
                vocabulary[entry]=wposTokens
    print(len(vocabulary))
    print(vocabulary)
    synsetDict={}
    for word in vocabulary: 
        wpos = word.split(".")
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
        pseudo = "none"
        synsetPos.append(pseudo)
        print(synsetPos)
        i=1
        for ss in synsetPos:
            if(i!=len(synsetPos)):
                print(str(i)+" "+ss.definition())
            else:
                print(str(i)+" "+ss)
            i=i+1
        print("context:")
        print(vocabulary[word])
        print("enter your choice:")
        choice = int(input())
        if(choice==len(synsetPos)):
            synsetDict[word]="none"
        else:
            tss = synsetPos[choice-1]
            nameofSynset = tss.name()
            synsetDict[word] = nameofSynset
    with open("senseMappingsGantt.txt","wb") as fp:
        pickle.dump(synsetDict,fp)

def taggerCheck():
    senseMappingsFile = filedialog.askopenfilename(initialdir=os.getcwd(),
                                                   title="Select Sense Mappings File")
    with open(senseMappingsFile,"rb") as fp:
        senseMappings = pickle.load(fp)
    
    print(len(senseMappings))
    tagged= [mapping for mapping in senseMappings if senseMappings[mapping]!='none']
    print(len(tagged))
    untagged = [mapping for mapping in senseMappings if mapping not in tagged]
    print(len(untagged))
    print(untagged)


def threadDivisionPoints(features,divisions):
        n1=0
        itr=0
        fl = len(features)
        ops= (fl*(fl-1))/2
        sps = [(0,1)]
        load= ops/divisions
        while(n1<len(features)-1):
            n2=n1+1
            while(n2<=len(features)-1):
                itr+=1
                n2+=1
                if(itr%load ==0):
                    if(n2!=len(features)-1):
                        if(n2!=len(features)):
                            sps.append((n1,n2))
                    else:
                        sps.append((n1+1,n2))
                    
            n1+=1
        print("Thread division points")
        if((len(features)-2,len(features)-1) not in sps):
            sps.append((len(features)-2,len(features)-1))
        print(sps)
        return sps
    
        
def computeSrs(features):
    n1=0
    itr = 0
    while(n1 < len(features)-1):
        n2 = n1+1
        while(n2 <=len(features)-1):
            itr+=1
            print("Iteration " + str(itr
                                     ))
            srDict[itr] = wnUtils.sr(wn.synset(features[n1]),wn.synset(features[n2]))
            n2+=1
        n1+=1
    
    return srDict

    
    
def multiThreadSrs(features,start,end,tid,divLenght):
    n1=start[0]
    n2 = start[1]
    itr=tid*divLenght    #div lenght represents work load for each thread
                         # tid should be equal to zero for first thread and 1 for next
                         # and so on
    while(n1 <= end[0]):
        if(n1!=start[0]):
            n2 = n1+1
        if(n1 ==end[0]):
            while(n2<=end[1]-1):
                itr+=1
            print("Tid "+ str(tid)+" ,Iteration"+str(itr))
            srDict[itr]=wnUtils.sr(wn.synset(features[n1]),wn.synset(features[n2]))
            n2+=1
        else:    
            while(n2<=len(features)-1):
                itr+=1
                print("Tid "+str(tid)+" ,Iteration"+str(itr))
                srDict[itr]=wnUtils.sr(wn.synset(features[n1]),wn.synset(features[n2]))
                n2+=1
        n1+=1
    
    
    
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
    print("doc mapping type "+ type(documentMapping).__name__)
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
    
# algorithm n(n-1)/2 dimensions
def rfs():
    ppdata = dataPreProcessing(InputFileNames,OutputFileNames,False)
    tfIdfs = getTfIdf(ppdata[0],ppdata[1])
    return tfIdfs

def cosineSim2():
    # includeUT is 3rd positional argument false for this alg
    ppdata = dataPreProcessing(InputFileNames,OutputFileNames,False)
    tfIdfs = getTfIdf(ppdata[0],ppdata[1])
    features = tfIdfs[2]
    print(features)
    rowsHigh = len(InputFileNames)
    rowsLow = len(OutputFileNames)
    cols = len(features)
    higher = np.ndarray((rowsHigh,cols))   #moon 
    lower =  np.ndarray((rowsLow,cols))    #moon
    idx= 0 # 2 matrices, 1 10x120 for high,5x120 1 for low level reqs,
    for item in tfIdfs:
        if(isinstance(item,list)):
            break    # break when features are read
        if(idx ==0):
            higher= item.todense()
        else:
            lower= item.todense()
            
        idx+=1
    
    similarity_matrix1  = cosine_similarity(higher,lower)
    similarities1 = []
    hi=0
    while hi<rowsHigh:         #moon
        li=0
        temp=[]
        while li<rowsLow:      #moon
            temp.append(similarity_matrix1[hi][li])
            li+=1
        hi+=1
        similarities1.append(temp)
    sortedSims1=[]
    for sim in similarities1:
        temp= [i for i in sorted(enumerate(sim), key=lambda x:x[1],reverse=True)]
        sortedSims1.append(temp)
    

    #Thread Divison Points
    #Threading code
    """
    workers= 10
    divisions=threadDivisionPoints(features,workers)
    
    load = ((len(features)*(len(features)-1))/2)/workers
    print("Load "+str(load))

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        idx=0
        while idx<=len(divisions)-2:
            executor.submit(multiThreadSrs,features,divisions[idx],divisions[idx+1],
                            idx,load)
            idx+=1
     """       
    #Threading code
    #here after obtaining the divisions it would be best to call the thread pool
    #executor
    #srs=computeSrs(features) can rest for a while, while we access srDict.
    """
    with open("srDict.txt","wb") as fp:
        pickle.dump(srs,fp)
    """
    Cols2 = (cols*(cols-1))/2
    with open("srDict.txt","rb") as fp:
        srs = pickle.load(fp)
    
    #print(len(tfIdfsDense[1]))
    tfIdfs2Higher = np.ndarray((rowsHigh,Cols2)) #moon# 2 matrices, 1 10x7140 for high, 5x7140 for low level reqs
    tfIdfs2Lower  = np.ndarray((rowsLow,Cols2))  #moon

    
    di=0
    for req in higher:
            pairs= []
            n1=0
            while n1<len(features)-1:
                n2 = n1+1
                while n2<=len(features)-1:
                    htfIdf = req.item(n1)+req.item(n2)
                    pairs.append(htfIdf)
                    n2+=1
                n1+=1
            pa = np.asarray(pairs)
            pa = pa.reshape(1,Cols2)     #moon
            tfIdfs2Higher[di]= pa #np.append(tfIdfs2Higher,pa,axis=0)
            di+=1
    di=0
    for req in lower:
            pairs= []
            n1=0
            while n1<len(features)-1:
                n2 = n1+1
                while n2<=len(features)-1:
                    htfIdf = req.item(n1)+req.item(n2)
                    pairs.append(htfIdf)
                    n2+=1
                n1+=1
            pa = np.asarray(pairs)
            pa = pa.reshape(1,Cols2)     #moon
            tfIdfs2Lower[di]=pa  #np.append(tfIdfs2Lower,pa,axis=0) 
            di+=1
    pi =0
    ri =0
    while ri<rowsHigh:
        while pi<Cols2:      #moon
            tfIdfs2Higher[ri][pi]=tfIdfs2Higher[ri][pi]*srs[pi+1]
            pi+=1
    
        ri+=1
            
    pi=0
    ri=0
    while ri<rowsLow:                 #moon
        while pi<Cols2:
            tfIdfs2Lower[ri][pi]=tfIdfs2Lower[ri][pi]*srs[pi+1]
            pi+=1
    
        ri+=1
    
    similarity_matrix2=cosine_similarity(tfIdfs2Higher,tfIdfs2Lower) #10x5 row high doc, col low doc
    hi=0
    similarities = []
    while(hi<rowsHigh):       #moon
        li=0
        while(li<rowsLow):
            sim= similarity_matrix2[hi][li]
            li+=1
            similarities.append(sim)
        hi+=1
    
    similaritiesSep =[]
    hi=0
    while(hi<rowsHigh):       #MOON
        
        li=0
        temp=[]
        while(li<rowsLow):
            temp.append(similarities[(hi*5)+li])
            li+=1
        similaritiesSep.append(temp)
        hi+=1
    sortedSims =[]
    for sim in similaritiesSep:
        temp= [i for i in sorted(enumerate(sim), key=lambda x:x[1],reverse=True)]
        sortedSims.append(temp)
    pi=0
    for item,item1 in zip(sortedSims,sortedSims1):
        pi+=1
        print(pi)
        print(item)
        print(item1)
    #only considering semantic. Now generate links,how?
    return [sortedSims,sortedSims1]
    #computeSrs(features)

def omiotis():
    tfIds = rfs()
    with open('srDict.txt','rb') as fp:
        srs = pickle.load(fp)
    print(srs)
omiotis()

def linkPrediction20(sims):
    hid=0
    link=[]
    for ls in sims:
        for tup in ls:
            if(tup[1] >= 0.35):
                link.append((hid,tup[0]))
        
        hid+=1
    print(link)
    print(len(link))
    return link


def recallMetric(link):
    tps=0
    for l in traceLinks:
        if l in link:
            tps+=1
    recall = tps/len(traceLinks)
    print(recall)

def precisionMetric(link):
    tps = 0
    for l in link:
        if l in traceLinks:
            tps+=1
    precision = tps/len(link)
    print(precision)
    
def ensembleSimilarities(sims):
    ensembleSims=[]
    for lsSem,lsSyn in zip(sims[0],sims[1]):
        highList=[]
        for tup1 in lsSem:
            for tup2 in lsSyn:
                if(tup1[0]==tup2[0]):
                    tup = (tup1[0],(tup1[1]+tup2[1])/2)
                    highList.append(tup)
                    break
        ensembleSims.append(highList)
    return ensembleSims
    
def datasetTests():        
    sims =cosineSim2()   
    semLinks=linkPrediction20(sims[0]) #Tuples with high id,low id
    synLinks=linkPrediction20(sims[1]) 
    
    print("semantic recall")
    recallMetric(semLinks)
    print("syntatic recall")
    recallMetric(synLinks)
    print("semantic precision")
    precisionMetric(semLinks)
    print("syntatic precision")
    precisionMetric(synLinks)
    
    ensemble = ensembleSimilarities(sims)
    ensLinks=linkPrediction20(ensemble)
    
    print("ensemble recall")
    recallMetric(ensLinks)
    print("ensemble precision")
    precisionMetric(ensLinks)
    print(ensLinks)
    print(len(ensLinks))

    #should be 120*119/2 for n = 120, 7140
#wordnetGraph():
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


# Global Variables

