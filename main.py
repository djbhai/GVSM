# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 21:43:28 2019

@author: User
"""
from tracing import tracing
from tracingMetrics import tracingMetrics
from os import listdir
import os
import re
import pickle
import matplotlib.pyplot as plt

#import pickle



def stripText(files):
    thresholdFiles={}
    for file in files:
        mo= re.search('[0-9.]+',file)
        threshold= mo.group(0)
        threshold = threshold[0:len(threshold)-1]
        threshold = float(threshold)
        thresholdFiles[threshold]= file
    return thresholdFiles

def getLinks(method):
    recalls=[]
    precisions=[]
    print("Enter dataset name")
    dataset = str(input())
    directory = "links"+method+dataset
    files=listdir(directory)
    trfiles =stripText(files)
    metrics = tracingMetrics()
    
    initial =0.0
    while initial<=1.0:
        file=trfiles[initial]
        path = directory+"/"+file
        with open(path,'rb') as fp:
            links=pickle.load(fp)
        print(links)
        recall=metrics.recallMetric(links)
        recalls.append(recall)
        precision=metrics.precisionMetric(links)
        precisions.append(precision)
        initial=round(initial+0.05,2)
    print(trfiles)
    return [recalls,precisions]
 
def plotGraph():
    data1= getLinks("Omiotis")
    data2= getLinks("VSM")
    recalls1 = data1[0]
    precisions1 =  data1[1]
    recalls2 = data2[0]
    precisions2 = data2[1]
    idx=0
    temp=[]
    for recall,precision in zip(recalls1,precisions1):
        if(recall==0.0):
            temp.append(idx)
        idx+=1
    i=0
    sortedTemp1= [i for i in sorted(temp,reverse=True)]
    while i < len(temp):
        recalls1.pop(sortedTemp1[i])
        precisions1.pop(sortedTemp1[i])
        i+=1
        
    idx=0
    temp=[]
    for recall,precision in zip(recalls2,precisions2):
        if(recall==0.0):
            temp.append(idx)
        idx+=1
    
    i=0
    sortedTemp2 = [i for i in sorted(temp,reverse=True)]

    while i<len(temp):
        recalls2.pop(sortedTemp2[i])
        precisions2.pop(sortedTemp2[i])
        i+=1
    print(recalls2)
    print(precisions2)
    plt.step(recalls1,precisions1,'s',color='orange')
    plt.step(recalls2,precisions2,'s',color='green')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.show()
"""
plotGraph()
"""    

"""
tracer = tracing()
initial = 0
while initial<=1.0:
    tracer.omiotisTest(initial)
    tracer.datasetTests(initial)
    initial = round(initial+0.05,2)
"""
"""
#tracer.omiotisTest()
#tracer.omiotisVSM()
"""