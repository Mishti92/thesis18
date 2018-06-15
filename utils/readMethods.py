#Import files
import numpy as np
import re
import sys
from Bio import SeqIO
from matplotlib import pyplot as plt
from itertools import product
import math
import sklearn
import sklearn.ensemble
from sklearn.model_selection import train_test_split
from sklearn import svm
from pprint import pprint
import csv
import pandas as pd
import random
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix,classification_report
from random import randint
import pickle

padding = True
newSizeList=[]
reduction = True

def readBedFile(filename, content):
    with open(filename)as f:
        for line in f:
            content.append(line.strip().split())
    return content

def sizeOfReads(content,size):
    for item in range(len(content)):
        upperRead = int(content[item][2])
        lowerRead = int(content[item][1])
        size.append(upperRead - lowerRead)
    return size

def plot_range(size):
    plt.hist(size)
    plt.title("Range of Binding Motifs")
    plt.xlabel("Length")
    plt.ylabel("Frequency")
    plt.show()
    
def dividePosNegLists(content,posList,negList):
    for item in range(len(content)):
        if content[item][5] == '+':
            posList.append(content[item])
        else:
            negList.append(content[item])
    return posList,negList

def writeToFileAndPadding(theList, theFile, padding, reduction, SeqList):
    thefile = open(theFile, 'w')
    for item in range(len(theList)):
        seqSpec = theList[item][0]
        lowerRead = int(theList[item][1])
        upperRead = int(theList[item][2])
        size = upperRead - lowerRead
#        print "Before size", upperRead - lowerRead
#         print size
        if size < 100:
            padding = (100 - size)/2
            lowerRead = lowerRead - padding
            upperRead = upperRead + padding
#            print "Padding", upperRead - lowerRead
#         newSizeList.append(upperRead - lowerRead)
        else:
#     print "yes"
            reduction = (size -100)/2
            lowerRead = lowerRead + reduction
            upperRead = upperRead - reduction
            
#print "After Reducing", upperRead - lowerRead
        seq=(seqSpec+":"+str(lowerRead)+"-"+str(upperRead))
        SeqList.append(seq)
        thefile.write(seq+"\n")
    thefile.close()
    return SeqList

def readfaFileToDict(theFileName,DNASeqList):
    handle = open(theFileName,"rU")
    i=0
    for seq_record in SeqIO.parse(handle, "fasta"):
        DNASeqList[seq_record.id]=seq_record.seq.tostring() 
    return DNASeqList



def DNAtoRNA(posDNASeqList,negDNASeqList,RNASeqList):
    rep = {"T": "a", "t":"a","A":"u","a":"u","C":"g","c":"g","G":"c","g":"c"}
    for key in posDNASeqList:
        RNASeqList[key]=posDNASeqList[key].replace("T","u").replace("t","u").replace("A","a").replace("C","c").replace("G","g")
    for key in negDNASeqList: 
        rep = dict((re.escape(k), v) for k, v in rep.iteritems())
        pattern = re.compile("|".join(rep.keys()))
        RNASeqList[key] = pattern.sub(lambda m: rep[re.escape(m.group(0))], negDNASeqList[key])
    return RNASeqList
