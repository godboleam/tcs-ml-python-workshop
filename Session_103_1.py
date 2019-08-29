#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#session_103_1.py:
"""
@author: Godbole

                                                                                                                                                                                                                                 

"""
#import sys
from pandas import Series, DataFrame
import pandas as pd
import re
import math #for logarithm function
#import random #required for randomizing a list
#import time #to measure time
#from collections import deque#3 https://docs.python.org/2/tutorial/datastructures.html  Required a fringe is a queue for BFS

#trainingfile,testfile,outputfile= str(argv[1],sys.argv[2],sys.argv[2])

trainingfile='tweets.train.clean.txt'
testfile='tweets.test1.clean.txt'
outputfile='output.txt'

def preprocess(wordlist):
    pwordlist=wordlist
    plowerwordlist=[]
    for word in pwordlist:
        word0=re.sub('[#,*,?,!,(,),&,.,:,;,/,-]', '', word)#https://stackoverflow.com/questions/13437114/how-do-i-replace-a-character-in-a-string-with-another-character-in-python
        word1=re.sub('[_,","]', ' ', word0)
        plowerwordlist.append(word1.lower())
    #excludelist=[]
    excludelist=[' ','@','at','the','in','to','you','as','i','my','me','this','that','those','here','there','a','is','was','but','on','of','and','for']
    for excludeword in excludelist:
        if excludeword in plowerwordlist:
            plowerwordlist.remove(excludeword)
    return plowerwordlist

def findclasses():#preporcessing not required for extracting the cities/classes from the training file
    citylist=[] #list of cities or classes, city may be repeated in this specific list
    totalnumtweets=0
    #trngfile=open(trainingfile,'r') # https://stackoverflow.com/questions/23917729/switching-to-python-3-causing-unicodedecodeerror
    trngf=open(trainingfile,'r') #Entire Training Document and names of Classess i.e. Entire training tweets and Cities
    ctraintweets=[]
    totalnumtweets=0 # Total numer of training documents or tweets 
    for ctweet in trngf:
        ctraintweets.append(ctweet)
        words=ctweet.split()  
        citylist.append(words[0])
        totalnumtweets+=1
    cityseries=pd.Series(citylist)
    citycount=cityseries.value_counts()
    #print(totalnumtweets,listofcities) # 32K tweets; 12 cities
    trngf.close()
    return(citycount,totalnumtweets)

def train(citycount,N):        
    trngf1=open(trainingfile,'r') #Entire Training Document and names of Classess i.e. Entire training tweets and Cities
    ctraintweets=[]
    for ctweet in trngf1:
        ctraintweets.append(ctweet)
    trngf1.close()
     
    #Building city-wise word distribution
    cityworddist={}
    for city in citycount.index:
        cityvocabulary=[]
        for ctweet in ctraintweets:
            ctweetsp=ctweet.split()
            if city==ctweetsp[0]:
                cityvocabulary+=preprocess(ctweetsp)#class name also included in training; classname is an associated word!; classname not considered while evaluating test file 
        cityvocabseries=pd.Series(cityvocabulary)
        cityvocabwordcount=cityvocabseries.value_counts()
        cityworddist[city]=cityvocabwordcount
        #print('******************')
        #print(city,cityvocabulary,cityvocabseries.value_counts())
        #print('******************')
    #print(cityworddist)
    
    prior={} #Prior Probability
    for city in citycount.index:
        prior[city]=citycount[city]/N #P(city)        
    return prior,cityworddist    
    
def numofwordintraining(): #preprocessing used at the end of this function!!!
    trngf1=open(trainingfile,'r') #Entire Training Document and names of Classess i.e. Entire training tweets and Cities
    ctraintweets=[]
    for ctweet in trngf1:
        ctraintweets.append(ctweet)
    trngf1.close()
    num=0
    vocab=[]
    for ctweet in ctraintweets:
        for word in ctweet.split()[1:]:
            if word not in vocab:
                vocab.append(word)
                num+=1    
    return preprocess(vocab),num #vocab has only distinct words

            
def processtestfile(citycount,prior,cityworddist,vocabsize,vocab):                    
    trngf3=open(testfile,'r') #Entire Training Document and names of Classess i.e. Entire training tweets and Cities
    testf=open(outputfile,'w')
    ctesttweets=[]
    for ctweet in trngf3:
        ctesttweets.append(ctweet)
    trngf3.close()
    TP=0
    numoftesttweets=0
    for ctweet in ctesttweets:
        numoftesttweets+=1
        scores={}
        z=0
        for city in citycount.index:
            score=math.log10(prior[city])
            #print(city,prior[city])
            #for word in preprocess(ctweet.split()):#preprocessing the test tweet; class name is also made lower case
            for word in preprocess(ctweet.split()[1:]):#preprocessing the test tweet; class name is excluded    
                if z==0:
                    actualcity=ctweet.split()[0] 
                    z+=1
                if word in cityworddist[city].index: 
                    a1=cityworddist[city][word]
                else:
                    a1=0
                a2=1
                a3=cityworddist[city].sum()
                a4=vocabsize
                #print(city,word,'a4',a4)
                score+=math.log10(a1+a2)-math.log10(a3+a4) #similar results obtained by using multiplication and without logarithms
                #if word not in vocab:
                    #score-=math.log10(prior[city])/len(ctweet.split()[1:]) #This is to adjust the prior probability for words not in the entire training set...this does not help if the test set distribution
                    #mirrors the distribution in the training set. However, this could help otherwise.
            scores[city]=score
            #print(city,scores[city])
        #print('*************')
        #print(scores)
        #print('*************')
        v=list(scores.values())
        k=list(scores.keys())
        predcity=k[v.index(max(v))]
        predv=max(v)
        #print(predcity,actualcity,predv,ctweet) #see this for tweet level predictions
        testf.write(predcity+' '+ctweet) 
        if predcity.lower()==actualcity.lower():
            TP+=1
    accuracy=100*TP/numoftesttweets
    print('True Positives=',TP,'Accuracy=',accuracy,'%')
    #testf.write('TP='+str(TP)+' Accuracy='+str(accuracy)+'%') #syntax needs to be changed after importing re
            
def citytopwords(citycount,cityworddist,n):
    print('Top words asscociated with each city:')
    for city in citycount.index:
        print(city,'\n',cityworddist[city][:n])
        
        
def main():
    citycount,N=findclasses()
    vocab,vocabsize=numofwordintraining()
    #print(citycount,N)
    prior,cityworddist=train(citycount,N)
    processtestfile(citycount,prior,cityworddist,vocabsize,vocab)
    citytopwords(citycount,cityworddist,5)
    
main()
