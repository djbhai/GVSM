# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 21:09:41 2019

@author: User
"""

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
import math
from spellchecker import SpellChecker
import networkx as nx
import matplotlib.pyplot as plt
import pprint

class wordnetUtils:
    """
    wn =0
    wnGraph = 0
    """
    def __init__(self,wn,wnGraph,pp):
        self.wn=nltk.corpus.wordnet
        self.wnGraph = self.wnGraphInitializer()
        self.pp = pprint.PrettyPrinter(width=41,compact=True)
        self.pp.pprint(self.wnGraph)   #debug
    #self can only be used in class methods scope
    #instance variables can be accessed using self in method scope.
    
    def symmetrify(self,wnGraph):
        # hypernym and hyponym are treated as same type of edges.So doesn't
        #matter if we assign the reverse edge as same type
        #while symmetrifying 
        toAdd = []
        doneEdges = set()
        for edge in wnGraph.edges():
            if(edge not in doneEdges):
                doneEdges.add(edge)
                adict = wnGraph[edge[0]][edge[1]]
                for key,value in adict.items():
                    rel = value['relation'] 
                
                    if((edge[1],edge[0]) in wnGraph.edges()):
                        odict =wnGraph[edge[1]][edge[0]]
                        condi = False
                        for key,value in odict.items():
                            if(value['relation'] ==rel):
                                condi = True
                        if(condi == False):
                            toAdd.append((edge[1],edge[0],rel))
                    else:
                        toAdd.append((edge[1],edge[0],rel))
                    
                    
        for tup in toAdd:
            wnGraph.add_edge(tup[0],tup[1],relation=tup[2])
    
    def actualedgeWeighting(self,wnGraph):
        hypeCount = 0
        instHypeCount = 0
        memMeroCount = 0
        partMeroCount = 0
        subsMeroCount = 0
        antoCount = 0
        pertCount = 0
        entailCount = 0
        attributeCount=0
        causesCount = 0
        alsoSeesCount = 0
        topDomainCount = 0
        regDomainCount = 0
        usageDomainCount = 0
        verbGroupCount =0
        derCount = 0
        similartoCount =0
        edgesCount =0
        nodes = wnGraph.nodes()
        for nd in nodes:
            outEdges = wnGraph.out_edges(nd)
            doneEdges = set()
            for edge in outEdges:
                if(edge not in doneEdges):
                    doneEdges.add(edge)
                    adict = wnGraph[edge[0]][edge[1]]
                    for key,value in adict.items():
                        edgesCount+=1
                        if((value['relation']=='hyponym' or value['relation']=='hypernym')):
                            hypeCount+=1
                        if((value['relation']=='inst_hyponym' or value['relation']=='inst_hypernym')):
                            instHypeCount+=1
                        if((value['relation']=='mem_meronym' or value['relation']=='mem_holonym')):
                            memMeroCount+=1
                        if((value['relation']=='part_meronym' or value['relation']=='part_holonym')):
                            partMeroCount+=1
                        if((value['relation']=='sub_meronym' or value['relation']=='sub_holonym')):
                            subsMeroCount+=1
                        if(value['relation']=='antonym'):
                            antoCount+=1
                        if(value['relation']=='pertainym'):
                            pertCount+=1
                        if(value['relation']=='entailment'):
                            entailCount+=1
                        if(value['relation']=='attribute'):
                            attributeCount+=1
                        if(value['relation']=='causes'):
                            causesCount+=1
                        if(value['relation']=='also_see'):
                            alsoSeesCount+=1
                        if(value['relation']=='top_domain'):
                            topDomainCount+=1
                        if(value['relation']=='reg_domain'):
                            regDomainCount+=1
                        if(value['relation']== 'use_domain'):
                            usageDomainCount+=1
                        if(value['relation']=='verb_group'):
                            verbGroupCount+=1
                        if(value['relation']=='derivational'):
                            derCount+=1
                        if(value['relation']=='similar_to'):
                            similartoCount+=1
                    
        hypeWeight = hypeCount/edgesCount
        instHypeWeight = instHypeCount/edgesCount
        memMeroWeight = memMeroCount/edgesCount
        partMeroWeight = partMeroCount/edgesCount
        subsMeroWeight = subsMeroCount/edgesCount
        antoWeight = antoCount/edgesCount
        pertWeight = pertCount/edgesCount
        entailWeight = entailCount/edgesCount
        attributeWeight = attributeCount/edgesCount
        causesWeight = causesCount/edgesCount
        alsoSeesWeight = alsoSeesCount/edgesCount
        topDomainWeight = topDomainCount/edgesCount
        regDomainWeight = regDomainCount/edgesCount
        usageDomainWeight = usageDomainCount/edgesCount
        verbGroupWeight = verbGroupCount/edgesCount
        derWeight = derCount/edgesCount
        similartoWeight = similartoCount/edgesCount
        for nd in nodes:
            outEdges = wnGraph.out_edges(nd)
            doneEdges = set()
            for edge in outEdges:
                if(edge not in doneEdges):
                    doneEdges.add(edge)
                    adict = wnGraph[edge[0]][edge[1]]
                    for key,value in adict.items():
                        if((value['relation']=='hyponym' or value['relation']=='hypernym')):
                            value['weight'] = hypeWeight
                        if((value['relation']=='inst_hyponym' or value['relation']=='inst_hypernym')):
                            value['weight'] = instHypeWeight
                        if((value['relation']=='mem_meronym' or value['relation']=='mem_holonym')):
                            value['weight'] = memMeroWeight
                        if((value['relation']=='part_meronym' or value['relation']=='part_holonym')):
                            value['weight'] = partMeroWeight
                        if((value['relation']=='sub_meronym' or value['relation']=='sub_holonym')):
                            value['weight'] = subsMeroWeight
                        if(value['relation']=='antonym'):
                            value['weight'] = antoWeight
                        if(value['relation']=='pertainym'):
                            value['weight'] = pertWeight
                        if(value['relation']=='entailment'):
                            value['weight'] = entailWeight
                        if(value['relation']=='attribute'):
                            value['weight'] = attributeWeight
                        if(value['relation']=='causes'):
                            value['weight'] = causesWeight
                        if(value['relation']=='also_see'):
                            value['weight'] = alsoSeesWeight
                        if(value['relation']=='top_domain'):
                            value['weight'] = topDomainWeight
                        if(value['relation']=='reg_domain'):
                            value['weight'] = regDomainWeight
                        if(value['relation']== 'use_domain'):
                            value['weight'] = usageDomainWeight
                        if(value['relation']=='verb_group'):
                            value['weight'] = verbGroupWeight
                        if(value['relation']=='derivational'):
                            value['weight'] = derWeight
                        if(value['relation']=='similar_to'):
                            value['weight'] = similartoWeight
        """
        print("weights:")
        print(hypeWeight)
        print(instHypeWeight)
        print(memMeroWeight)
        print(partMeroWeight) 
        print(subsMeroWeight)
        print(antoWeight)
        print(pertWeight)
        print(entailWeight)
        print(attributeWeight)
        print(causesWeight)
        print(alsoSeesWeight)
        print(topDomainWeight)
        print(regDomainWeight)
        print(usageDomainWeight)
        print(verbGroupWeight)
        print(derWeight)
        print(similartoWeight)
        """
        return wnGraph
    
    def wordnetGraph(self):
        G = nx.MultiDiGraph()
        alls = self.wn.all_synsets()
        for a in alls:
            G.add_node(a,depth=a.max_depth()+1)
        
            hyponyms=a.hyponyms()
            instance_hyponyms=a.instance_hyponyms()
            hypernyms=a.hypernyms()
            instance_hypernyms= a.instance_hypernyms()
            member_holonyms = a.member_holonyms()
            substance_holonyms=a.substance_holonyms()
            part_holonyms = a.part_holonyms()
            member_meronyms = a.member_meronyms()
            substance_meronyms=a.substance_meronyms()
            part_meronyms = a.part_meronyms()
            attributes =  a.attributes()
            entailments = a.entailments()
            verb_groups = a.verb_groups()
            causes = a.causes()
            also_sees = a.also_sees()
            similar_tos = a.similar_tos()
            
            if(len(hyponyms)!=0):
                for hyponym in hyponyms:
                    G.add_edge(a,hyponym,relation="hyponym")
            if(len(instance_hyponyms)!=0):
                for instance_hyponym in instance_hyponyms:
                    G.add_edge(a,instance_hyponym,relation="inst_hyponym")
            if(len(hypernyms)!=0):
                for hypernym in hypernyms:
                    G.add_edge(a,hypernym,relation="hypernym")
            if(len(instance_hypernyms)!=0):
                for instance_hypernym in instance_hypernyms:
                    G.add_edge(a,instance_hypernym,relation="inst_hypernym")
            if(len(member_holonyms)!=0):
                for member_holonym in member_holonyms:
                    G.add_edge(a,member_holonym,relation="mem_holonym")
            if(len(substance_holonyms)!=0):
                for substance_holonym in substance_holonyms:
                    G.add_edge(a,substance_holonym,relation="sub_holonym")
            if(len(part_holonyms)!=0):
                for part_holonym in part_holonyms:
                    G.add_edge(a,part_holonym,relation="part_holonym")
            if(len(member_meronyms)!=0):
                for member_meronym in member_meronyms:
                    G.add_edge(a,member_meronym,relation="mem_meronym")
            if(len(substance_meronyms)!=0):
                for substance_meronym in substance_meronyms:
                    G.add_edge(a,substance_meronym,relation="sub_meronym")
            if(len(part_meronyms)!=0):
                for part_meronym in part_meronyms:
                    G.add_edge(a,part_meronym,relation="part_meronym")
            if(len(attributes)!=0):
                for attribute in attributes:
                    G.add_edge(a,attribute,relation="attribute")
            if(len(entailments)!=0):
                for entailment in entailments:
                    G.add_edge(a,entailment,relation="entailment")
            if(len(verb_groups)!=0):
                for verb in verb_groups:
                    G.add_edge(a,verb,relation="verb_group")
            if(len(causes)!=0):
                for cause in causes:
                    G.add_edge(a,cause,relation="causes")
            if(len(also_sees)!=0):
                for also_see in also_sees:
                    G.add_edge(a,also_see,relation="also_see")
            if(len(similar_tos)!=0):
                for similar_to in similar_tos:
                    G.add_edge(a,similar_to,relation="similar_to")
                    
            #Lemma relations antonyms,pertainyms,derivationally_related forms,domains
            lemmas = a.lemmas()
            fl = lemmas[0]
            antonyms= fl.antonyms()
            if(len(antonyms)>0):
                for antonym in antonyms:
                    ans = antonym.synset()
                    G.add_edge(a,ans,relation="antonym")
            for lemma in lemmas:
                 drf = lemma.derivationally_related_forms()
                 if(len(drf)!=0):
                     for dr in drf:
                         drs = dr.synset()
                         G.add_edge(a,drs,relation="derivational")
            for lemma in lemmas:
                topic_domains= lemma.topic_domains()
                region_domains= lemma.region_domains()
                usage_domains = lemma.usage_domains()
                if(len(topic_domains)!=0):
                    for l in topic_domains:
                        tds = l.synset()
                        G.add_edge(a,tds,relation="top_domain")
                if(len(region_domains)!=0):
                    for l in region_domains:
                        rds = l.synset()
                        G.add_edge(a,rds,relation="reg_domain")
                if(len(usage_domains)!=0):
                    for l in usage_domains:
                        uds = l.synset()
                        G.add_edge(a,uds,relation="use_domain")
            for lemma in lemmas:
                pertainyms = lemma.pertainyms()
                if(len(pertainyms)!=0):
                    for p in pertainyms:
                        ps = p.synset()
                        G.add_edge(a,ps,relation="pertainym")
        
        self.symmetrify(G)  
        self.actualedgeWeighting(G)
        #symmetrifying the graph by including bidirectional edges for all relations
        #except hypernym,hyponym,holonym,meronym
    
        """
        with open("WordnetGraph.txt","wb") as fp:
            write_gpickle
        """
        print("I am here")
        print(len(G.edges()))   #debug
        return G

    def wnGraphInitializer(self):            
        return self.wordnetGraph()
        
    
    #wordnetUtils.wnGraphInitializer=staticmethod(wordnetUtils.wnGraphInitializer)
    
    
    
    def expander(self,currentPass,subGraph,leftSubGraph,rightSubGraph):
    
         leftPass= currentPass[0]
         rightPass =currentPass[1]
         nextLeft = set()
         nextRight = set()
         condition = True
         for item in leftPass:
             expand = self.wnGraph.neighbors(item)
             for x in expand:
                 if x in rightSubGraph:
                     condition = False
                     #return False
                 if x not in subGraph:
                     nextLeft.add(x)
                     subGraph.add(x)
                     leftSubGraph.add(x)
         for item in rightPass:
            expand = self.wnGraph.neighbors(item)
            for x in expand:
                if x in leftSubGraph:
                    condition = False
                    #return False
                if x not in subGraph:
                    nextRight.add(x)
                    subGraph.add(x)
                    rightSubGraph.add(x)
         
         return [nextLeft,nextRight,condition]
                
    
    def computeNetwork(self,s1,s2):
        left = {s1}
        right= {s2}
        currentPass = [left,right,True]
        leftSubGraph = {s1}
        rightSubGraph = {s2}
        subGraph = {s1,s2}
        noCon = False
        while(currentPass[2]):
            if((len(currentPass[0]) or len(currentPass[1]))==0):
                noCon = True
                break
            currentPass=self.expander(currentPass,subGraph,leftSubGraph,rightSubGraph)
        if(noCon == True):
            return 0
            #may be return true or Zero
        connection = self.wnGraph.subgraph(subGraph)
        return connection
    
    
    
    def splitter(string):
        splits =string.split("_")
        return splits
        
    def edgeWeighting(self,connection):
        nodeCount = connection.number_of_nodes()
        hypeCount = 0
        meroCount = 0
        antoCount = 0
        pertCount = 0
        entailCount = 0
        attributeCount=0
        causesCount = 0
        alsoSeesCount = 0
        domainCount = 0
        verbGroupCount =0
        derCount = 0
        similartoCount =0
        nodes = connection.nodes()
        
        
        for nd in nodes:
            outEdges = connection.out_edges(nd)
            hype = False
            mero = False
            anto = False
            pert = False
            entail = False
            attribute = False
            causes =False
            alsoSees = False
            domain = False
            verbG = False
            der = False
            sim = False
            doneEdges = set()
            for edge in outEdges:
                if(edge not in doneEdges):
                    doneEdges.add((edge[0],edge[1]))
                    adict=connection[edge[0]][edge[1]]
                    for key,value in adict.items():
                        splits = self.splitter(value['relation'])
                        if(splits.contains('hyponym') or splits.contains('hypernym')): #change it to contains
                            hype = True
                        if(splits.contains('meronym') or splits.contains('holonym')):  #change it to contains
                            mero = True
                        if(value['relation']=='antonym'):
                            anto = True
                        if(value['relation']=='pertainym'):
                            pert = True
                        if(value['relation']=='entailment'):
                            entail = True
                        if(value['relation']=='attribute'):
                            attribute = True
                        if(value['relation']=='causes'):
                            causes = True
                        if(value['relation']=='also_see'):
                            alsoSees = True
                        if(splits.contains('domain')): #change it to contains
                            domain = True
                        if(value['relation']=='verb_group'):
                            verbG = True
                        if(value['relation']=='derivational'):
                            der = True
                        if(value['relation']=='similar_to'):
                            sim = True
            if(hype == True):
                hypeCount+=1
            if(mero == True):
                meroCount+=1
            if(anto == True):
                antoCount+=1
            if(pert == True):
                pertCount+=1
            if(entail == True):
                entailCount+=1
            if(attribute==True):
                attributeCount+=1
            if(causes == True):
                causesCount+=1
            if(alsoSees == True):
                alsoSeesCount+=1
            if(domain == True):
                domainCount+=1
            if(verbG == True):
                verbGroupCount+=1
            if(der == True):
                derCount+=1
            if(sim == True):
                similartoCount+=1
        invHype =0
        invMero =0
        invAnto =0
        invPert =0
        invEnt = 0
        invAtt = 0
        invCaus = 0
        invAs = 0
        invDom = 0
        invVG = 0
        invDer = 0
        invSim =0
        
    
        if(hypeCount!=0):            
            invHype = math.log10((nodeCount+1)/hypeCount)
        if(meroCount!=0):            
            invMero = math.log10((nodeCount+1)/meroCount)
        if(antoCount!=0):            
            invAnto = math.log10((nodeCount+1)/antoCount)
        if(pertCount!=0):
            invPert = math.log10((nodeCount+1)/pertCount)
        if(entailCount!=0):            
            invEnt  = math.log10((nodeCount+1)/entailCount)
        if(attributeCount!=0):
            invAtt  = math.log10((nodeCount+1)/attributeCount)
        if(causesCount!=0):
            invCaus = math.log10((nodeCount+1)/causesCount)
        if(alsoSeesCount!=0):
            invAs   = math.log10((nodeCount+1)/alsoSeesCount)
        if(domainCount!=0):
            invDom  = math.log10((nodeCount+1)/domainCount)
        if(verbGroupCount!=0):
            invVG   = math.log10((nodeCount+1)/verbGroupCount)
        if(derCount!=0):
            invDer  = math.log10((nodeCount+1)/derCount)
        if(similartoCount!=0):
            invSim  = math.log10((nodeCount+1)/similartoCount)
        
        print("hype inv")
        print(invHype)
        print("end")
        #return subGraph
        for nd in nodes:
            outEdges = connection.out_edges(nd)
            hypeCount = 0
            meroCount = 0
            antoCount = 0
            pertCount = 0
            entailCount = 0
            attributeCount=0
            causesCount = 0
            alsoSeesCount = 0
            domainCount = 0
            verbGroupCount =0
            derCount = 0
            similartoCount =0
            
            doneEdges = set()
            for edge in outEdges:
                if(edge not in doneEdges):
                    doneEdges.add((edge[0],edge[1]))
                    adict=connection[edge[0]][edge[1]]
                    for key,value in adict.items():
                        if(splits.contains('hyponym') or splits.contains('hypernym')): #change it to contains
                            hypeCount+=1
                        if(splits.contains('meronym') or splits.contains('holonym')):  #change it to contains
                            meroCount+=1
                        if(value['relation']=='antonym'):
                            antoCount+=1
                        if(value['relation']=='pertainym'):
                            pertCount+=1
                        if(value['relation']=='entailment'):
                            entailCount+=1
                        if(value['relation']=='attribute'):
                            attributeCount+=1
                        if(value['relation']=='causes'):
                            causesCount+=1
                        if(value['relation']=='also_see'):
                            alsoSeesCount+=1
                        if(splits.contains('domain')): #change it to contains
                            domainCount+=1
                        if(value['relation']=='verb_group'):
                            verbGroupCount+=1
                        if(value['relation']=='derivational'):
                            derCount+=1
                        if(value['relation']=='similar_to'):
                            similartoCount+=1
            allWeight = (hypeCount+meroCount+antoCount+pertCount+entailCount)+(attributeCount+causesCount+alsoSeesCount+domainCount+verbGroupCount)+(derCount+similartoCount)
            hypeWeight = hypeCount/allWeight
            meroWeight = meroCount/allWeight
            antoWeight = antoCount/allWeight
            pertWeight = pertCount/allWeight
            entailWeight = entailCount/allWeight
            attributeWeight = attributeCount/allWeight
            causesWeight = causesCount/allWeight
            alsoSeesWeight = alsoSeesCount/allWeight
            domainWeight  = domainCount/allWeight
            verbGroupWeight = verbGroupCount/allWeight
            derWeight = derCount/allWeight
            similarWeight = similartoCount/allWeight     
                        
            doneEdges= set()       
            for edge in outEdges:
                if(edge not in doneEdges):
                    doneEdges.add((edge[0],edge[1]))
                    adict=connection[edge[0]][edge[1]]
                    for key,value in adict.items():
                        if(splits.contains('hyponym') or splits.contains('hypernym')): #change it to contains
                            value['weight'] = hypeWeight*invHype
                        if(splits.contains('meronym') or splits.contains('holonym')):  #change it to contains
                            value['weight'] = meroWeight*invMero
                        if(value['relation']=='antonym'):
                            value['weight'] = antoWeight*invAnto
                        if(value['relation']=='pertainym'):
                            value['weight'] = pertWeight*invPert
                        if(value['relation']=='entailment'):
                            value['weight'] = entailWeight*invEnt
                        if(value['relation']=='attribute'):
                            value['weight'] = attributeWeight*invAtt
                        if(value['relation']=='causes'):
                            value['weight'] = causesWeight*invCaus
                        if(value['relation']=='also_see'):
                            value['weight'] = alsoSeesWeight*invAs
                        if(splits.contains('domain')): #change it to contains
                            value['weight'] = domainWeight*invDom
                        if(value['relation']=='verb_group'):
                            value['weight'] = verbGroupWeight*invVG
                        if(value['relation']=='derivational'):
                            value['weight'] = derWeight*invDer
                        if(value['relation']=='similar_to'):
                            value['weight'] = similarWeight*invSim
        return connection
       
                
    
    def symRelationsTest(self,rel):
        symmetric = True
        relCount=0
        for edge in self.wnGraph.edges():
        
            adict =self.wnGraph[edge[0]][edge[1]]
            for key,value in adict.items():
                if(value['relation']==rel):
                    relCount+=1
                    pt = False
                    for pred in self.wnGraph.predecessors(edge[0]):
                        if(pred ==edge[1]):
                            pt = True
                            break
                    if(pt==False):
                        symmetric= False
                        print(edge)
                        print(relCount)
                        return symmetric
                    if(pt==True):
                        odict=self.wnGraph[edge[1]][edge[0]]
                        condition = False
                        for key,value in odict.items():
                            if(value['relation']==rel):
                                relCount+1
                                condition= True
                        if(condition==False):
                            symmetric =False
                            print(edge)
                            print(relCount)
                            print(odict)
                            return symmetric
        return symmetric
    
    
    def scaling(weighted):
        weightList = []
        doneEdges =set()
        for edge in weighted.edges():
            if(edge not in doneEdges):
                adict = weighted[edge[0]][edge[1]]
                doneEdges.add((edge[0],edge[1]))
                for key,value in adict.items():
                    weigh =value['weight'] 
                    weightList.append(weigh)
    
        
        maximum = max(weightList)
        minimum = min(weightList)
        print(maximum)
        print(minimum)
        print("range")
        height = maximum - minimum
        doneEdges = set()
        if(height !=0):
            for edge in weighted.edges():
                if(edge not in doneEdges):
                    doneEdges.add((edge[0],edge[1]))
                    adict = weighted[edge[0]][edge[1]]
                    for key,value in adict.items():
                        updDict = {(edge[0],edge[1],key):{'weight':(value['weight']-minimum)/height}}
                        nx.set_edge_attributes(weighted,updDict)
                    #weighted[edge[0]][edge[1]][key]['weight']= (value['weight']-minimum)
            
                            
            """
            for edge in weighted.edges():        
                adict = weighted[edge[0]][edge[1]]
                for key,value in adict.items():
                    value['weight'] /= height
                    #weighted[edge[0]][edge[1]][key]['weight'] =value['weight']/height
            """
        return weighted
    
        
    def scaledWeightedGraph(self,s1,s2):
        
        sg =self.computeNetwork(s1,s2)
        if(sg ==0):
            print("There is no connection between the synsets")
            return 0
        else:
            
            weightList = []
            weighted = self.edgeWeighting(sg)
            scaled = self.scaling(weighted)
            """
            for edge in scaled.edges():
                
                adict = scaled[edge[0]][edge[1]]
                for key,value in adict.items():
                    scaled[edge[0]][edge[1]][key]['weight'] =value['weight']
                    """
            """
             weightList  = []
            for edge in scaled.edges():
                adict =scaled[edge[0]][edge[1]]
                for key,value in adict.items():
                    weightList.append(value['weight'])
            print(max(weightList))
            print(min(weightList))
            """
            return scaled
    
    
    def maxDepth(self):
        depList=[]
        for node in self.wnGraph.nodes():
            depList.append(self.wnGraph.nodes[node]['depth'])
        
        print(max(depList))
    
    
                
    def srComputation(self,s1,s2):
        found = set()
        visited = set()
        visited.add(s1)
        toexpand = s1
        network = self.computeNetwork(s1,s2) #incase other method needs to be used network should be replaced 
                                        #by scaled and the below lines of code must be used.
        #scaled = scaledWeightedGraph(s1,s2)
        if(network == 0):
            return 0
        nx.set_node_attributes(network,0,'sr')
        #print("SR initialization Test")
        #print(scaled.nodes[s2]['sr'])
        insrDict ={toexpand:{'sr':1}}
        nx.set_node_attributes(network,insrDict)
        """
        print("s1 depth")
        """
        d1= network.nodes[s1]['depth']
        """
        print(d1)
        """
        if(network == 0):
            return 0
        else:
            while(s2 not in visited):
                successors=set(network.neighbors(toexpand))
                for item in successors-visited:
                    ts = network.nodes[toexpand]['sr']
                    adict = network[toexpand][item]
                    iweight =0
                    for key,value in adict.items():
                        if(iweight<value['weight']):
                            iweight = value['weight']
                    d1= network.nodes[toexpand]['depth']
                    d2 = network.nodes[item]['depth']
                    num = 2*d1*d2
                    den = (d1+d2)*20
                    hm = num/den
                    sr = hm*iweight*ts
                    if(sr>network.nodes[item]['sr']):
                        srDict = {item:{'sr':sr}}
                        nx.set_node_attributes(network,srDict)
                        found.add(item)
                maxim =0
                for item in found:
                    if(network.nodes[item]['sr']>maxim):
                        maxim =network.nodes[item]['sr']
                        maximNd = item
                found.remove(maximNd)
                """
                print("maxNd "+str(maximNd))
                print(maxim)
                """
                #print("max found "+str(maxim))
                visited.add(maximNd)
                toexpand = maximNd
               
        self.pp.pprint("sr of the second synset")       
        self.pp.pprint(network.nodes[s2]['sr'])  
        """    
        print("number of edges")
        print(nx.number_of_edges(network))
        """
        return network.nodes[s2]['sr']
    
                        
    def sr(self,s1,s2):
        sr1= self.srComputation(s1,s2)
        """
        sr2 = srComputation(s2,s1)
        print("srs:")
        print(sr1)
        print(sr2)
        srVal = (sr1+sr2)/2
        print(srVal)
        
        return srVal
        """
        return sr1
    
    """
    s1 = wn.synset("basenji.n.01")
    s2 = wn.synset("dog.n.01")
    sr(s1,s2)
    """   
        
        
    #print(len(sg))
    def test():
        #print(wn.synset('entity.n.01'))
        #wn.synset('abstraction.n.06')
        #test for symmetry of relations
        #symmetrify(wnGraph)
        
        
        """
        b = symRelationsTest('pertainym')
        print(b)
        """
        """
        for edges in wnGraph.out_edges(wn.synset('fall_for.v.02')):
            print(edges[1])
        adict = wnGraph[edges[0]][edges[1]]
        for key,value in adict.items():
            print(value)
    
        for pred in wnGraph.predecessors(wn.synset('fall_for.v.02')):
            print(pred)
        print(sg.out_edges(wn.synset('fall_for.v.02')))
        print(sg.in_degree(wn.synset('fall_for.v.02')))
        for pred in sg.predecessors(wn.synset('fall_for.v.02')):
            print(pred)
    
        print(sg[wn.synset('fall.v.03')][wn.synset('fall_for.v.02')])
        """
        """
        
        ss= wn.synset('attribute.n.02')
        print(ss.hyponyms())
        print(ss.instance_hyponyms())
        print(ss.hypernyms())
        print(ss.member_holonyms())
        print(ss.substance_holonyms())
        print(ss.part_holonyms())
        print(ss.member_meronyms())
        print(ss.substance_meronyms())
        print(ss.part_meronyms())
        print(ss.entailments())
        print(ss.verb_groups())
        ds = wn.synset('dog.n.01')
        print(ds.root_hypernyms())
        
        i=0
        for s in wn.all_synsets('v'):
            if i<10:
                print(s)
                i=i+1
            else:
                break
        """
        """
        G= nx.DiGraph()
        G.add_node("s1",depth=15)
        G.add_node("s2")
        G.add_edge("s1","s2",relation='hypernym')
        print(G.nodes['s1']['depth'])
        G.add_nodes_from([1,2,3])
        G.add_edges_from([(1, 2, {'color': 'blue','depth':(0,1)}), (2, 3, {'weight': 8})])
        plt.subplot(121)
        nx.draw(G,with_labels=True, font_weight='bold')
        print(G["s1"]["s2"]['relation'])
        plt.show()
        
        #vss = wn.all_synsets('v')
        
        sam = wn.synset('book.n.01')
        samls = sam.lemmas()
        for saml in samls:
            print(saml.pertainyms())
            ps = saml.pertainyms()
            for p in ps:
               print( p.synset())
        s= wn.synset('direct.v.01')
        lemmas = s.lemmas()
        print(lemmas)
        if(len(lemmas)!=0):
            print(lemmas[0].derivationally_related_forms())
            for lem in lemmas[0].derivationally_related_forms():
                print(lem.synset())
            
            a= lemmas[0].antonyms()
        print(a)
        if(len(a)>0):
            print(a[0].synset())
            
             
        s = wn.synset('president_of_the_united_states.n.01')
        print(s.hyponyms())
        ss= wn.synset('act.v.01')
        print(ss.root_hypernyms())
         """      
