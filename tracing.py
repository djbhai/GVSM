# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 19:37:29 2019

@author: User
"""
import wordnetUtils
import nltk.corpus
import numpy as np
from scipy.sparse import csr_matrix
from os import listdir
import os
import sys
import signal
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from nltk.corpus import stopwords
from tkinter import filedialog
import re
import pickle
from spellchecker import SpellChecker
import concurrent.futures
import gc

class tracing:
    
    def __init__(self):
        
        
        self.wnUtils = wordnetUtils.wordnetUtils()
        self.wn= nltk.corpus.wordnet
        self.If = filedialog.askdirectory(initialdir=os.getcwd(),title="select high directory")
        print(self.If)
        self.Of = filedialog.askdirectory(initialdir=os.getcwd(),title="select low directory")
        print(self.Of)
        self.InputFileNames= listdir(self.If)
        self.OutputFileNames= listdir(self.Of)
        self.srDict={}      #To store sr's in index numbered dict
        self.Vocab =set([])
        self.Untagged = set([])
        self.inputDict={}
        self.outputDict={}
        self.synsetDict={}
        print("Does sense Mapping's file exist for the dataset you provided?y/n")
        smo = str(input())
        vocabulary=dict()
        entered= set()
        stopWords = set(stopwords.words('english'))
        tokenizedDict  = self.tokenize(self.InputFileNames,self.OutputFileNames)
        for key in tokenizedDict:
            wposTokens=tokenizedDict[key]
            for token in wposTokens:
                if(token[0].lower() not in stopWords):
                    if(re.search('\A^[a-zA-Z0-9]',token[0])):
                        entry=  token[0].lower()+"."+token[1]
                        if(len(vocabulary)!=0):
                            entered=[item for item in vocabulary if item==entry ]
                        if(len(entered)==0):
                            vocabulary[entry]=wposTokens
        if(smo=='y'):
        #if exists load otherwise compute using wordnetTagger
            self.senseMappingsFile = filedialog.askopenfilename(initialdir=os.getcwd(),
                                                   title="Select Sense Mappings File")
            with open(self.senseMappingsFile,"rb") as fp:
                self.senseMappings = pickle.load(fp)
            self.synsetDict = self.senseMappings
            print("len of senseMappings"+str(len(self.senseMappings)))
        
            if(len(self.senseMappings)==len(vocabulary)):
                pass
            else:
                self.senseMappings = self.wordnetTagger(vocabulary,self.senseMappings)
    
                
        else:
           
            self.senseMappings = self.wordnetTagger(vocabulary)
            if(len(self.senseMappings)==len(vocabulary)):
                pass
            else:
                self.senseMappings = self.wordnetTagger(vocabulary,self.senseMappings)
        
        
        print("Does  srDict inclusive file exist for the dataset you provided?y/n")
        srio = str(input())
        if(srio=='y'):
             self.srDictInc = filedialog.askopenfilename(initialdir=os.getcwd(),title=
                                                    "select inclusive sr dictionary file")
            
        else:
            self.storeSrs(True)
            self.srDictInc =  filedialog.askopenfilename(initialdir=os.getcwd(),title=
                                                    "select inclusive sr dictionary file")
        
        """
        print("Does  srDict exclusive file exist for the dataset you provided?y/n")
        sreo =str(input())
        if(sreo=='y'):
            self.srDictEx =  filedialog.askopenfilename(initialdir=os.getcwd(),title=
                                                    "select exclusive sr dictionary file")
        else:
            self.srDictEx =  self.storeSrs(False)
        #if not compute both sr dicts.
        """
        
        #self.traceLinks =[(1,0),(3,1),(4,2),(5,0),(7,3),(9,4),(6,3),(7,4)]  #moon lander
        
        print("enter the trace matrix file")
        self.tmFile=filedialog.askopenfilename(initialdir=os.getcwd(),title=
                                      "select trace matrix")
        self.traceLinks=self.rtm(self.tmFile)
        print("enter dataset name")
        self.dataset = str(input())
        
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
    
    def Tokenizer(self,s):
            regexp = '(\w)*\.[avrns]\.[0-9]*|(\w)*\.[avrns]'
            wds = s.split(" ")
            return wds
    
    def get_wordnet_pos(self,listOfTokens):
        """Map POS tag to first character lemmatize() accepts"""
        tag_dict = {"J": self.wn.ADJ,
                    "N": self.wn.NOUN,
                    "V": self.wn.VERB,
                    "R": self.wn.ADV}
        posTokens= nltk.pos_tag(listOfTokens)
        wposTokens = []
        for token in posTokens:
            tag = token[1][0][0].upper()
            tag= tag_dict.get(tag,self.wn.NOUN)
            wposTokens.append((token[0],tag))  
        return wposTokens
    
    
    
    def tokenize(self,inputs,outputs):
        pathToInputFiles = []
        pathToOutputFiles  = []
        tokenizedDict={}
        lemmatizer = WordNetLemmatizer()
        for file in inputs:
            pathToInputFiles.append(self.If+"/"+file)
        
        for index,file in zip(inputs,pathToInputFiles):
            fp = open(file,'r')
            content = fp.read()       
            
            tokens = nltk.word_tokenize(content)
            wposTokens=[(lemmatizer.lemmatize(w[0],w[1]),w[1]) for w in self.get_wordnet_pos(tokens)]
            tokenizedDict[index]=wposTokens
        
        return tokenizedDict
    
    
    def dataPreProcessing(self,inputs,outputs,includeUT):
        pathToInputFiles = []
        pathToOutputFiles  = []
        lemmatizer = WordNetLemmatizer()
        totalTokens =0
        stop_words = set(stopwords.words('english'))   
        wposDict = self.tokenize(inputs,outputs)
    
        
        for index,sentence in wposDict.items():
            woSwords = []
    
            for token in sentence:
                if(token[0].lower() not in stop_words):
                    if(re.search('\A^[a-zA-Z0-9]',token[0])):
                            #totalTokens+=1
                            #This code needs to be changed as none tokens
                            #AREN'T added
                        if(token[0].lower()+'.'+token[1] in self.senseMappings):
                            self.Vocab.add(token[0].lower()+'.'+token[1])
                            totalTokens+=1
                            if(includeUT ==True):
                                if(self.senseMappings[token[0].lower()+'.'+token[1]]=='none'):
                                    self.Untagged.add(token[0].lower()+'.'+token[1])
                                    woSwords.append(token[0].lower()+'.'+token[1])
                                else:
                                    woSwords.append(self.senseMappings[token[0].lower()+'.'+token[1]])
                            else:
                                if(self.senseMappings[token[0].lower()+'.'+token[1]]!='none'):
                                    woSwords.append(self.senseMappings[token[0].lower()+'.'+token[1]])
    
                                    
                        elif(includeUT):
                            self.Untagged.add(token[0].lower()+'.'+token[1])
                            woSwords.append(token[0].lower()+'.'+token[1])
                        list(filter(('none').__ne__, woSwords))
                        self.inputDict[index]=woSwords
        for file in outputs:
            pathToOutputFiles.append(self.Of+"/"+file)
        for index,file in zip(outputs,pathToOutputFiles):
            fp = open(file,'r')
            content = fp.read()
            tokens = nltk.word_tokenize(content)
            wposTokens=[(lemmatizer.lemmatize(w[0],w[1]),w[1]) for w in self.get_wordnet_pos(tokens)]
            woSwords = []
            for token in wposTokens:
                if(token[0].lower() not in stop_words):
                    if(re.search('\A^[a-zA-Z0-9]',token[0])):
                        if(token[0].lower()+'.'+token[1] in self.senseMappings):
                            self.Vocab.add(token[0].lower()+'.'+token[1])
                            totalTokens+=1
                            if(includeUT==True):
                                if(self.senseMappings[token[0].lower()+'.'+token[1]]=='none'):
                                    woSwords.append(token[0].lower()+'.'+token[1])
                                else:
                                    woSwords.append(self.senseMappings[token[0].lower()+'.'+token[1]])
                            else:
                                if(self.senseMappings[token[0].lower()+'.'+token[1]]!='none'):
                                     woSwords.append(self.senseMappings[token[0].lower()+'.'+token[1]])
                                
                        elif(includeUT):
                            woSwords.append(token[0].lower()+'.'+token[1])
                        
    
                        #totalTokens+=1
                        list(filter(('none').__ne__, woSwords))
                        self.outputDict[index]=woSwords
        ppdata =[self.inputDict,self.outputDict]
        #print(senseMappings)
        return ppdata
    
    
    
    """
    entered= set()
    tokenizedDict  = self.tokenize(self.InputFileNames,self.OutputFileNames)
     for key in tokenizedDict:
            wposTokens=tokenizedDict[key]
            for token in wposTokens:
                entry=  token[0]+"."+token[1]
                if(len(vocabulary)!=0):
                    entered=[item for item in vocabulary if item==entry ]
                if(len(entered)==0):
                    vocabulary[entry]=wposTokens
            
    """
    #def keyBoardInterruptHandler(self,signal,frame):
        
    def vsmPreProcessing(self,inputs,outputs):
        pathToInputFiles = []
        pathToOutputFiles  = []
        inputTokenizedDict = {}
        outputTokenizedDict ={}
        lemmatizer = WordNetLemmatizer()
        totalTokens =0
        stop_words = set(stopwords.words('english'))
        for file in inputs:
            pathToInputFiles.append(self.If+"/"+file)
        
        for index,file in zip(inputs,pathToInputFiles):
            fp = open(file,'r')
            content = fp.read()
            tokens = nltk.word_tokenize(content)
            lemTokens=[lemmatizer.lemmatize(w[0],w[1]) for w in self.get_wordnet_pos(tokens)]
            inputTokenizedDict[index]= lemTokens
        for index,sentence in inputTokenizedDict.items(): 
            finalTokens=[]
            for token in sentence:
                if(token.lower() not in stop_words):
                    if(re.search('\A^[a-zA-Z0-9]',token.lower())):
                        finalTokens.append(token.lower())
            self.inputDict[index] = finalTokens
        for file in outputs:
            pathToOutputFiles.append(self.Of+"/"+file)
        for index,file in zip(outputs,pathToOutputFiles):
            fp = open(file,'r')
            content = fp.read()
            tokens = nltk.word_tokenize(content)
            lemTokens=[lemmatizer.lemmatize(w[0],w[1]) for w in self.get_wordnet_pos(tokens)]
            outputTokenizedDict[index]= lemTokens
        for index,sentence in outputTokenizedDict.items():
            finalTokens=[]
            for token in sentence:
                if(token.lower() not in stop_words):
                    if(re.search('\A^[a-zA-Z0-9]',token.lower())):
                        finalTokens.append(token.lower())
            self.outputDict[index] = finalTokens
        ppdata =[self.inputDict,self.outputDict]
        return ppdata
            
     
    def vsmTfIdf(self):
        inputs =[]
        outputs=[]
        vocab=dict({})
        i=0
        for key in self.inputDict:
            inputString=""
            j=0
            for word in self.inputDict[key]:
                    if(word!='none'):
                        if(j+1 == len(self.inputDict[key])):
                            inputString+=word
                        else:
                            inputString+=word+" "
                            j=j+1
                        if(word not in vocab):
                            vocab[word]=i
                            i=i+1
            inputs.append(inputString)
    
        for key in self.outputDict:
            outputString=""
            j=0
            for word in self.outputDict[key]:
                if(word!='none'):
                    if(j+1 == len(self.outputDict[key])):
                            outputString+=word
                    else:
                        outputString+=word+" "
                        j=j+1
            outputs.append(outputString)
        vectorizer = TfidfVectorizer(input='content',norm='l2',vocabulary=vocab)
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
        
        

        
    def wordnetTagger(self,vocabulary,senseMappings=None):
        #should add code to use it as convinience function and remove tagging left overs.
        print("enter dataset name")
    
        choice = str(input())
        filename = "senseMappings"+choice+".txt"
        done = set()
        if(senseMappings ==None):
            pass
        else:
            for key in senseMappings:
                done.add(key)
        spell = SpellChecker()
        print(len(vocabulary))
        print(vocabulary)
        #signal.signal(signal.SIGINT,self.keyBoardInterruptHandler)
        idx=0
        for word in vocabulary:
            idx+=1
            print("words complete")
            print(idx)
            if word not in done:
                wpos = word.split(".")
                wd =wpos[0]
                pos = wpos[1]
                synsetPos = []
                changedSynsetPos=[]
                synset =[]
                if(self.wn.synsets(wd)):
                    print(wd)
                    synset = self.wn.synsets(wd)
                
                for ss in synset:
                    if(ss.pos()==pos):
                        synsetPos.append(ss)
                    elif(ss.pos()=='s' and pos=='a'):
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
                choice = input()
                if(choice=='exit'):
                     with open(filename,"wb") as fp:
                         pickle.dump(self.synsetDict,fp)
                     sys.exit()
                     
                else:
                    
                    choice =int(choice)
                    if(choice==len(synsetPos)):
                        self.synsetDict[word]="none"
                        print("enter s to change spelling"+
                              " and enter pos choice"+ 
                              ",p to just change the pos")
                        opC = str(input())
                        if(opC == 's'):
                           correct = spell.correction(wd)
                           correctPosSynsets=[]
                           print("Corrected word " + correct)
                           synset = self.wn.synsets(correct)
                           posc = ['a','n','v','r']
                            
                           print("enter your pos choice")
                           for item in posc:
                               print(item)
                           pose = str(input())
                           for ss in synset:
                               if(ss.pos()==pose):
                                   correctPosSynsets.append(ss)
                               elif(ss.pos()=='s' and pose=='a'):
                                   correctPosSynsets.append(ss)
                           correctPosSynsets.append("none")
                           i=1
                           for ss in correctPosSynsets:
                               if(i!= len(correctPosSynsets)):
                                   print(str(i)+" "+ss.definition())
                               else:
                                   print(str(i)+" "+ss)
                               i=i+1
                           print("context:")
                           print(vocabulary[word])
                           print("Enter your choice")
                           cpChoice = int(input())
                           if(cpChoice == len(correctPosSynsets)):
                               self.synsetDict[word]="none"
                           else:
                               tss = correctPosSynsets[cpChoice-1]
                               nameofSynset = tss.name()
                               self.synsetDict[word]=nameofSynset
                        
                        elif(opC=='p'):
                            posc = ['a','n','v','r']
                            
                            print("enter your pos choice")
                            for item in posc:
                                print(item)
                            pose = str(input())
                            for ss in synset:
                                if(ss.pos()==pose):
                                    changedSynsetPos.append(ss)
                                elif(ss.pos()=='s' and pose=='a'):
                                    changedSynsetPos.append(ss)
                            pseudo = "none"
                            changedSynsetPos.append(pseudo)
                            i=1
                            for ss in changedSynsetPos:
                                if(i!=len(changedSynsetPos)):
                                    print(str(i)+" "+ss.definition())
                                else:
                                    print(str(i)+" "+ss)
                                i=i+1
                            print("context:")
                            print(vocabulary[word])
                            print("enter your choice:")
                            newChoice=input()
                            if(newChoice == 'exit'):
                                with open(filename,"wb") as fp:
                                    pickle.dump(self.synsetDict,fp)
                                sys.exit()
                            else:
                                newChoice =int(newChoice)
                                if(newChoice==len(changedSynsetPos)):
                                    self.synsetDict[word] ="none"
                                else:
                                    tss = changedSynsetPos[newChoice-1]
                                    nameofSynset = tss.name()
                                    self.synsetDict[word] = nameofSynset
                                
                        
                    
                    else:
                        tss = synsetPos[choice-1]
                        nameofSynset = tss.name()
                        self.synsetDict[word] = nameofSynset
        
        
        with open(filename,"wb") as fp:
            pickle.dump(self.synsetDict,fp)
        return self.synsetDict
    

    
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
     
        
    def inclusiveSrs(self,features):
        n1=0
        itr = 0
        print("length of features")
        print(len(features))
        while(n1 <= len(features)-1):
            n2 = n1
            while(n2 <=len(features)-1):
                itr+=1
                print("Iteration" + str(itr))
                itTuple=(features[n1],features[n2])
                
                self.srDict[itTuple] = self.wnUtils.sr(features[n1],features[n2])
                n2+=1
            n1+=1
        return self.srDict
            
    def exclusiveSrs(self,features):
        srDict=dict()
        n1=0
        itr = 0
        while(n1 <= len(features)-1):
            n2 = n1
            while(n2 <=len(features)-1):
                itr+=1
                itTuple = (features[n1],features[n2])
                print("Iteration " + str(itr))
                srDict[itTuple] = self.wnUtils.sr(features[n1],features[n2])  #previous method of storing sr's using 
                #index used in cosineSim method,the method should be changed to facilitate 
                # tuple indexing
                n2+=1
            n1+=1
        return srDict

    
        
        
    def multiThreadSrs(self,features,start,end,tid,divLenght):
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
                self.srDict[itr]=self.wnUtils.sr(self.wn.synset(features[n1]),
                           self.wn.synset(features[n2]))
                n2+=1
            else:    
                while(n2<=len(features)-1):
                    itr+=1
                    print("Tid "+str(tid)+" ,Iteration"+str(itr))
                    self.srDict[itr]=self.wnUtils.sr(self.wn.synset(features[n1]),
                               self.wn.synset(features[n2]))
                    n2+=1
            n1+=1
        
        
        
    def getTfIdf(self):
        inputs =[]
        outputs=[]
        vocab=dict({})
        i=0
        for key in self.inputDict:
            inputString=""
            j=0
            for word in self.inputDict[key]:
                    if(word!='none'):
                        if(j+1 == len(self.inputDict[key])):
                            inputString+=word
                        else:
                            inputString+=word+" "
                            j=j+1
                        if(word not in vocab):
                            vocab[word]=i
                            i=i+1
            inputs.append(inputString)
    
        for key in self.outputDict:
            outputString=""
            j=0
            for word in self.outputDict[key]:
                if(word!='none'):
                    if(j+1 == len(self.outputDict[key])):
                            outputString+=word
                    else:
                        outputString+=word+" "
                        j=j+1
            outputs.append(outputString)
        #print(inputs)
        #print(outputs)
        print('len of vocab in getTfIdf'+ str(len(vocab)))
        
        vectorizer = TfidfVectorizer(input='content',norm='l2',vocabulary=vocab,
                                     token_pattern='((\w)*\.[avnsr]\.[0-9]*)',tokenizer=self.Tokenizer)
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
    
    def TfIdfTest(self):
        # algorithm n(n-1)/2 dimensions
        # includeUT is 3rd positional argument false
        ppdata = self.dataPreProcessing(self.InputFileNames,self.OutputFileNames,True)
        tfIdfs = self.getTfIdf(ppdata[0],ppdata[1])
        print(tfIdfs[2])
        print(len(tfIdfs[2]))
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
        print(tfIdfsDense[0].shape)
    
    def storeSrs(self,inclusive):
        if(inclusive == True):
            filename = "srDictInc"
        else:
            filename = "srDict"
        print("enter dataset")
        dataset =  str(input())
        filename = filename+dataset+".txt"
        features =[]
        for item in self.senseMappings:
            if(inclusive):
                if(self.senseMappings[item]=='none'):
                    features.append(item)
                else:
                    features.append(self.senseMappings[item])
            else:
                if(self.senseMappings[item] !='none'):
                    features.append(self.senseMappings[item])
        if(inclusive==True):
            srs = self.inclusiveSrs(features)
        else:
            srs = self.exclusiveSrs(features)
       
        
        with open(filename,"wb") as fp:
            pickle.dump(srs,fp)
        return srs
    
    #storeSrs(False)
        
    # algorithm n(n-1)/2 dimensions

    
    def cosineSims1(self):
        self.vsmPreProcessing(self.InputFileNames,self.OutputFileNames)
        tfIdfs = self.vsmTfIdf()
        features = tfIdfs[2]
        print(features)
        print(len(features))
        rowsHigh = len(self.InputFileNames)
        rowsLow = len(self.OutputFileNames)
        cols = len(features)
        higher = np.ndarray((rowsHigh,cols))   
        lower =  np.ndarray((rowsLow,cols))    
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
            
        return sortedSims1
        
    def cosineSim2(self):
            # includeUT is 3rd positional argument false for this alg
        self.dataPreProcessing(self.InputFileNames,self.OutputFileNames,False)
        tfIdfs = self.getTfIdf()
        features = tfIdfs[2]
        print(features)
        print(len(features))
        rowsHigh = len(self.InputFileNames)
        rowsLow = len(self.OutputFileNames)
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
        with open(self.srDictInc,"rb") as fp:      #Should be entered
            srs = pickle.load(fp)
        
        #print(len(tfIdfsDense[1]))
        tfIdfs2Higher = np.ndarray((rowsHigh,int(Cols2))) #moon# 2 matrices, 1 10x7140 for high, 5x7140 for low level reqs
        tfIdfs2Lower  = np.ndarray((rowsLow,int(Cols2)))  #moon
    
        
        di=0
        for req in higher:
                pairs= []
                n1=0
                while n1<len(features)-1:
                    n2 = n1+1
                    while n2<=len(features)-1:
                        try:
                            htfIdf = (req.item(n1)+req.item(n2))*srs[(features[n1],features[n2])]
                        except KeyError:
                            htfIdf = (req.item(n1)+req.item(n2))*srs[(features[n2],features[n1])]
                        pairs.append(htfIdf)
                        n2+=1
                    n1+=1
                pa = np.asarray(pairs)
                pa = pa.reshape(1,int(Cols2))    #moon
                tfIdfs2Higher[di]= pa #np.append(tfIdfs2Higher,pa,axis=0)
                di+=1
        di=0
        for req in lower:
                pairs= []
                n1=0
                while n1<len(features)-1:
                    n2 = n1+1
                    while n2<=len(features)-1:
                        try:
                            htfIdf = (req.item(n1)+req.item(n2))*srs[(features[n1],features[n2])]
                        except KeyError:
                            htfIdf = (req.item(n1)+req.item(n2))*srs[(features[n2],features[n1])]
                        pairs.append(htfIdf)
                        n2+=1
                    n1+=1
                pa = np.asarray(pairs)
                pa = pa.reshape(1,int(Cols2))     #moon
                tfIdfs2Lower[di]=pa  #np.append(tfIdfs2Lower,pa,axis=0) 
                di+=1
        print("cols2")
        print(Cols2)
        
        
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
        
        filename= "GVSM"+"simMat"+self.dataset+".txt"
        with open(filename,'wb') as fp:
            pickle.dump(sortedSims,fp)
        
        #only considering semantic. Now generate links,how?
        return sortedSims


    
        #computeSrs(features)
    #smf= filedialog.askopenfilename(initialdir=os.getcwd(),
    #    title="Select Sense Mappings File")   
    def senseMappingsLower(self,smf):
        lowSMF = dict()
        with open(smf,"rb") as fp:
            senseMappings = pickle.load(fp)
        stop_words = set(stopwords.words('english'))   
    
        for key in senseMappings:
            parts = key.split(".")
            if(parts[0].lower() in stop_words):
                continue
            else:
                lowSMF[key.lower()]= senseMappings[key].lower()
        outputfn="lowSenseMappings"
        print("enter dataset") 
        outputfn+=str(input())  +".txt"  
        with open(outputfn,'wb') as fp:
            pickle.dump(lowSMF,fp)
    #senseMappingsLower(smf)
    
    def omiotis(self):
        self.dataPreProcessing(self.InputFileNames,self.OutputFileNames,True) #True only needed for Ml
        tfIdfs = self.getTfIdf()
        features = tfIdfs[2]
        maxSim = 0
        minSim = 1
        with open(self.srDictInc,'rb') as fp:    #need to load the srDict by providing input
            srs = pickle.load(fp)
        
        """
        hd  = tfIdfs[0].todense()
        ld  = tfIdfs[1].todense()
        high = np.asarray(hd)
        low = np.asarray(ld)
        """
        high = tfIdfs[0]
        low = tfIdfs[1]
        h = np.shape(high)
        l= np.shape(low)
        hs = h[0]
        ls = l[0]
        #straight = np.ndarray((hs,ls,len(features),len(features)))
        #inverse = np.ndarray((ls,hs,len(features),len(features)))
        straight = np.ndarray((hs,ls),dtype=object)
        inverse = np.ndarray((ls,hs),dtype=object)
        
        #results = np.ndarray((hs*ls,))
        hi=0
        while hi< hs:
            li=0
            while li < ls:
                docRow = high[hi]
                queryRow = low[li]
                #dqdata = np.ndarray((len(features),len(features)))
                dqdata = []
                idx=0
                while idx < len(features):
                    dqdata.append([0])
                    idx+=1
                row =0
                ri=0
                while ri < len(docRow.data):
                    col=0
                    tfIdf1 = docRow.data[ri]
                    featureIndex1 = docRow.indices[ri]
                    lri=0
                    while lri <len(queryRow.data):
                        tfIdf2 = queryRow.data[lri]
                        featureIndex2 = queryRow.indices[lri]
            
                        try:
                            sr = srs[(features[featureIndex1],features[featureIndex2])]
                        except KeyError:
                            sr = srs[(features[featureIndex2],features[featureIndex1])]
                        
                        
                        if(dqdata[row][0]< ((2*tfIdf1*tfIdf2)/(tfIdf1+tfIdf2))*sr):
                            dqdata[row][0]=((2*tfIdf1*tfIdf2)/(tfIdf1+tfIdf2))*sr
                        lri+=1
                        col+=1
                    ri+=1
                    row+=1
                straight[hi][li] = dqdata 
                li+=1
            hi+=1
            print("Done "+ str(hi))
        
        straightSim = np.ndarray((hs,ls))
        hi=0
        while hi< hs:
            li=0
            while li < ls:
                row =0
                summation=0
                nnzero = 0
                """
                while row < len(straight[hi][li][len(features)-1]):
                    col=0
                    while col < len(straight[hi][li][len(features)-1]):
                        maximum = max(straight[hi][li][row])
                        if(maximum!=0):
                            summation+=maximum
                            nnzero+=1
                        
                        col+=1
                    row+=1
                """
                for mat in straight[hi][li]:
                    for row in mat:
                        if(row==0):
                            pass
                        else:
                            summation+=row
                            nnzero+=1
                            
                if(nnzero!=0):
                    sim = summation/nnzero
                    
                else:
                    sim = 0
                straightSim[hi][li] = sim
                
                        
                li+=1
            hi+=1    
    
        gc.collect()
        li=0
        while li <ls:
            hi = 0
            while hi <hs:
                queryRow = low[li]
                docRow = high[hi]
                #dqdata = np.ndarray((len(features),len(features)))
                dqdata = []
                idx=0
                while idx < len(features):
                    dqdata.append([0])
                    idx+=1
                row =0
                ri=0
                while ri < len(queryRow.data):
                    tfIdf1 = queryRow.data[ri]
                    featureIndex1 = queryRow.indices[ri]
                    col =0
                    lri = 0
                    while lri<len(docRow.data):
                        tfIdf2 = docRow.data[lri]
                        featureIndex2=docRow.indices[lri]
                    
                        try:
                            sr = srs[(features[featureIndex1],features[featureIndex2])]
                        except KeyError:
                            sr = srs[(features[featureIndex2],features[featureIndex1])]


                        #dqdata[row][col] = ((2*item*item2)/(item+item2))*(sr)
                        if(dqdata[row][0]< ((2*tfIdf1*tfIdf2)/(tfIdf1+tfIdf2))*sr):
                            dqdata[row][0]=((2*tfIdf1*tfIdf2)/(tfIdf1+tfIdf2))*sr
                        lri+=1
                        col+=1
                    ri+=1
                    row+=1
                inverse[li][hi] = dqdata
                hi+=1
            li+=1
            print("Done "+str(li))
        inverseSim = np.ndarray((ls,hs))
       
        li=0
        while li <ls:
            hi = 0
            while hi <hs:
                row =0
                summation=0
                nnzero=0
                """
                while row < len(inverse[li][hi][len(features)-1]):
                    col =0
                    while col < len(inverse[li][hi][len(features)-1]):
                        maximum = max(inverse[li][hi][row])
                        if(maximum!=0):
                            summation+=maximum
                            nnzero+=1
                        col+=1
                    row+=1
                """
                for mat in inverse[li][hi]:
                    for row in mat:
                            if(row==0):
                                pass
                            else:
                                summation+=row
                                nnzero+=1
                                
                
                if(nnzero!=0):
                    sim = summation/nnzero
                else:
                    sim=0
                inverseSim[li][hi] = sim
                hi+=1
            li+=1
        sim = np.ndarray((hs,ls))
        hi=0
        while hi<hs:
            li=0
            while li<ls:
                sim[hi][li] = (straightSim[hi][li]+inverseSim[li][hi])/2
                if(maxSim < sim[hi][li]):
                    maxSim = sim[hi][li]
                if(minSim > sim[hi][li]):
                    minSim = sim[hi][li]
                li+=1
            hi+=1

        flatSim = sim.flatten()
        flatSim = np.reshape(flatSim,(-1,1))
        scaler = MinMaxScaler()
        scaledSim = scaler.fit_transform(flatSim)
        sim = np.reshape(scaledSim,(hs,ls))
        
        simFile = "similarityMatrix"+self.dataset+".npy"
        with open(simFile,'wb') as fp:
            np.save(fp,sim)
        print("Minimum sim"+ str(minSim))
        print("Maximum sim"+str(maxSim))
        return sim
                      
            
        """
        with open('srDict.txt','rb') as fp:
            srs = pickle.load(fp)
        print(srs)
        """
    
     
    #file name to int
    def fn2Int(self):  
        idx=0
        fn2IntDict=dict()
        for name in self.InputFileNames:
            fn2IntDict[name]=idx
            idx+=1
        idx=0
        for name in self.OutputFileNames:
            fn2IntDict[name]=idx
            idx+=1
        return fn2IntDict
    
    def rtm(self,tmFile):
        fp = open(tmFile,'r')
        fn2IntDict=self.fn2Int()
        traceLinks=[]
        counter=0
        for item in fp:
            if(counter%2==0):
                counter+=1
                continue
            else:
                item=item.replace('\n','')
                #low=re.findall( '((\w)*\.txt)',item)
                low= re.findall( '([a-zA-Z0-9._-]+)',item)
                print(low)
                if(len(low)<=1):
                    continue
                highIx = fn2IntDict[low[0]]
                for req in low[1:len(low)]:
                    traceLinks.append((highIx,
                                       fn2IntDict[req]))
                    #traceLinks.append((highIx,fn2IntDict[req]))
        
            counter+=1
        print(traceLinks)
        print(len(traceLinks))
        return traceLinks
    """
    print("enter the trace matrix file")
    tmFile=filedialog.askopenfilename(initialdir=os.getcwd(),title=
                                      "select trace matrix")
    traceLinks=rtm(tmFile)
    """         
    def linkPrediction(self,sims,threshold):
        hid=0
        link=[]
        for ls in sims:
            for tup in ls:
                if(tup[1] >= threshold):
                    link.append((hid,tup[0]))
            
            hid+=1
        return link
    
    
    def recallMetric(self,link):
        tps=0
        for l in self.traceLinks:
            if l in link:
                tps+=1
    
        recall = tps/len(self.traceLinks)
    
        print(recall)
    
    def precisionMetric(self,link):
        tps = 0
        for l in link:
            if l in self.traceLinks:
                tps+=1
        if(len(link)==0):
            precision=1
        else:
            precision = tps/len(link)
        print(precision)
        
    def ensembleSimilarities(self,sims):
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
    
    def intersectionSimilarities(self,sims):
        return sims[0].intersection(sims[1])
        
    
    def omiotisTest(self,threshold):
        #sm = np.load("similarityMatrixGantt.npy")
        smFile = "similarityMatrix"+self.dataset+".npy"
        if(os.path.exists(smFile)):
            sm = np.load(smFile)
        else:
            sm = self.omiotis()
        isim=[]
        for item in sm:
            temp= [i for i in sorted(enumerate(item), key=lambda x:x[1],reverse=True)]
            isim.append(temp)
        links=self.linkPrediction(isim,threshold)
        linkDir = "links"+"Omiotis"+self.dataset
        file = linkDir+"/"+"links"+str(threshold)+".txt"
        if not os.path.exists(linkDir):
            os.mkdir(linkDir)
        with open(file,'wb') as fp:
            pickle.dump(links,fp)
        print("Omiotis recall:")
        self.recallMetric(links)
        print("Omiotis precision:")
        self.precisionMetric(links)
    
    #omiotisTest()
    
    def datasetTests(self,threshold):
        filenameGVSM= "GVSM"+"simMat"+self.dataset+".txt"
        if not os.path.exists(filenameGVSM):
            
            semSims=self.cosineSim2() #Tuples with high id,low id
        else:
            with open(filenameGVSM,'rb') as fp:
                semSims=pickle.load(fp)
        synSims=self.cosineSims1()
        semLinks=self.linkPrediction(semSims,threshold)
        synLinks = self.linkPrediction(synSims,threshold)
        semDir = "links"+"GVSM"+self.dataset
        synDir = "links"+"VSM"+self.dataset
        file = "links"+str(threshold)+".txt"
        semFile = semDir+"/"+file
        synFile = synDir+"/"+file
        if not os.path.exists(semDir):
            os.mkdir(semDir)
        if not os.path.exists(synDir):
            os.mkdir(synDir)
        with open(semFile,'wb') as fp:
            pickle.dump(semLinks,fp)
        with open(synFile,'wb') as fp:
            pickle.dump(synLinks,fp)
        
        print("VSM recall")
        self.recallMetric(synLinks)
        print("VSM precision")
        self.precisionMetric(synLinks)
        print("GVSM recall")
        self.recallMetric(semLinks)
        print("GVSM precision")
        self.precisionMetric(semLinks)
        sims = [semSims,synSims]
        ensemble = self.ensembleSimilarities(sims)
        ensLinks=self.linkPrediction(ensemble,0.2)
        
        print("ensemble recall")
        self.recallMetric(ensLinks)
        print("ensemble precision")
        self.precisionMetric(ensLinks)
        
    def omiotisVSM(self):
        vsmSims=self.cosineSims1()
        omiotisSims = self.omiotis()
        print("enter omiotis threshold")
        ot = float(input())
        isim=[]
        for item in omiotisSims:
            temp= [i for i in sorted(enumerate(item), key=lambda x:x[1],reverse=True)]
            isim.append(temp)
        olinks = set(self.linkPrediction(isim,ot))
        print("enter vsm threshold")
        vt = float(input())
        vlinks = set(self.linkPrediction(vsmSims,vt))
        ilinks = olinks.intersection(vlinks)
        print("Intersection recall")
        self.recallMetric(ilinks)
        print("Intersection precision")
        self.precisionMetric(ilinks)
        
        
    #datasetTests()
        #should be 120*119/2 for n = 120, 7140
    #wordnetGraph():
    #regexp is actually redundant as tokens are seperated by space
    def regexpTest():
        #reg = '((\w)*\.[avnsr]\.[0-9]*)|((\w)*\.[avnsr])'
        reg = '((\w)*\.txt)'
        sample = "r1.txt r2.txt"
        #some other regexp must be written to ensure that it get's all the vocab
        op = re.findall(reg,sample)
        print(op)
       
    
    
    
    def TaggingLeftOvers(self):
        self.dataPreProcessing(self.InputFileNames,self.OutputFileNames,True)
        posChanged = set()
        print(self.Untagged)
        print(len(self.Untagged))
        utVocContext = dict()
        entered = set()
        tokenizedDict  = self.tokenize(self.InputFileNames,self.OutputFileNames)
        for key,wposTokens in tokenizedDict.items():
            for token in wposTokens:
                entry=  token[0].lower()+"."+token[1]
                if((entry not in utVocContext) and (entry in self.Untagged)):
                    utVocContext[entry]=wposTokens
        print("utVocContext:")           
        print(len(utVocContext))
                
        # just load sensemappings,add mappings to taggable words.
        # insert in sensemappings dictionary and dump.
        
        pos = {self.wn.ADJ,self.wn.NOUN,self.wn.VERB,self.wn.ADV}
        spell=SpellChecker()
        leftOverDict =dict([])
        #vocabulary = Vocab
        #wordnetTagger(vocabulary)
        with open('senseMappingsGantt.txt','rb') as fp:
            l = pickle.load(fp)
        
        """
        leftOver = set([])
        for k in Vocab:
            if k not in l:
                leftOver.add(k)
        print(len(leftOver))
        print(leftOver)
        """
        for word in self.Untagged:
            lem= word.split(".")
            wd = lem[0]
            wdpos = lem[1]
            synsets=self.wn.synsets(wd)
            if(len(synsets)==0):
                correction=spell.correction(wd)
                print("change pos:")
                print("word")
                print(word)
                print(utVocContext[word])
                print("enter pos:")
                poses = ['a','n','v','r','x','l']
                pi =1
                for p in poses:
                    print(str(pi)+" "+p)
                    pi+=1
                posc = int(input())
                if(posc!=6):
                    wdpos = poses[posc-1]
                    if(posc!=5):
                        posChanged.add(word)
                synsets = self.wn.synsets(correction)
                posSynsets = []
                if(len(synsets)!=0):
                    for ss in synsets:
                        if(ss.pos() == wdpos):
                            posSynsets.append(ss)
                        elif(ss.pos =='s' and wdpos=='a'):
                            posSynsets.append(ss)
                            
                if(len(posSynsets)!=0):
                    i=1
                    print(correction)
                    print("enter your choice:")
                    for ss in posSynsets:
                        print(str(i)+ " "+ss.definition())
                        i=i+1
                    print(str(0)+" "+"continue")
                    choice = int(input())
                    if(choice == 0):
                        leftOverDict[word] = 'none'
                        continue
                    tss =posSynsets[choice-1]
                    name = tss.name()
                    leftOverDict[word] =name
                else:
                    leftOverDict[word] = 'none'
            else:
                i=1
                print(word)
                wordSplit = word.split(".")
                wdpos=  wordSplit[1]
                print("Context:")
                print(utVocContext[word])
                print("change pos")
                print("enter pos:")
                poses = ['a','n','v','r','x','l']
                pi =1
                for p in poses:
                    print(str(pi)+" "+p)
                    pi+=1
                posc = int(input())
                if(posc!=6):
                    wdpos = poses[posc-1]
                    if(posc!=5):
                        posChanged.add(word)
                posSynsets=[]
                for ss in synsets:
                    if(ss.pos() == wdpos):
                        posSynsets.append(ss)
                    elif(ss.pos() =='s' and wdpos=='a'):
                        posSynsets.append(ss)
                print("enter your choice:")
                if(posSynsets!=0):
                    for ss in posSynsets:
                        print(str(i)+ " "+ss.definition())
                        i=i+1
                else:
                    leftOverDict[word] = 'none'
                    continue
            
                print(str(0)+" "+"continue")
                choice = int(input())
                if(choice ==0):
                    continue
                tss =posSynsets[choice-1]
                name = tss.name()
                leftOverDict[word] =name
        with open('leftOverMappingsGantt.txt','wb') as fp:
            pickle.dump(leftOverDict,fp)
          
        
        with open('leftOverMappingsGantt.txt','rb') as fp:
            lo = pickle.load(fp)
        for mapp in lo:
            l[mapp]=lo[mapp]
        with open('senseMappingsGantt.txt','wb') as fp:
            pickle.dump(l,fp)
        print(len(l))
        print(len(lo))
        with open('senseMappingsGantt.txt','rb') as fp:
            l=pickle.load(fp)
        print(l)
    
    
    #wordnetTaggerTest()
    
    def test(self):
        op = self.getTfIdf(self.InputFileNames,self.OutputFileNames)
        for o in op:
            print(csr_matrix(o).toarray())
    
    def test2(self):
        self.wnUtils.srComputation('lion.n.01','cat.n.01')
    
    # Global Variables
    
