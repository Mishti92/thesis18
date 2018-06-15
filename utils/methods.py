#Import files
import numpy as np
import sys
from Bio import SeqIO
from matplotlib import pyplot as plt
from itertools import product
import math
from pprint import pprint
import csv
import pandas as pd
import random
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle
from itertools import islice
from random import shuffle

def readFile(theFileName):
    handle = open(theFileName,"rU")
    i=0
    seq_id=[]
    for seq_record in SeqIO.parse(handle, "fasta"):
        seq_id.append(seq_record.id)
    return seq_id

def readAndStorePWM(theFileName,seq_id,PWM):
    with open(theFileName)as f:
        for line in f:
#             print line
            for i in range(len(seq_id)):   
                if line.startswith('>'+seq_id[i]):
#                     print seq_id[i]
                    iterator = islice(f, 4)
                    pwm_a = next(iterator)
                    pwm_c = next(iterator)
                    pwm_g = next(iterator)
                    pwm_t = next(iterator)
#                     print pwm_a, pwm_c, pwm_g, pwm_t
                    PWM[seq_id[i]]={'a':pwm_a.split(),
                                    'c':pwm_c.split(),
                                    'g':pwm_g.split(),
                                    'u':pwm_t.split()}
    return PWM

def find_pwscore_for_rbp_for_nuc(rbp,nuc, PWM):
    return PWM[rbp][nuc]
    
def find_len_of_rbp(rbp,PWM):
    return len(PWM[rbp]['a'])

def cal_window_score(window_length_rbp, PWM, string,rbp):
#     print string, window_length_rbp
    score = 0
    for i in range(window_length_rbp):
#         print float(PWM[rbp][string[i]][i])
        score += float(PWM[rbp][string[i]][i])
    return score

def find_max_score_for_rbp(rbp,PWM,window_length_rbp):
    for nuc in PWM[rbp]:
        PWM[rbp][nuc] = [float(ele) for ele in PWM[rbp][nuc]]
    score = 0
    for i in range(window_length_rbp):
        score += max(PWM[rbp]['a'][i],PWM[rbp]['c'][i],PWM[rbp]['g'][i],PWM[rbp]['u'][i])
    return score

def stringWindowScore(inputSeq,window_length_rbp,max_score,PWM,rbp):
    stringScore = {}
    sizeSeq = len(inputSeq)
#     print sizeSeq
#     i = 0
    for i in range(sizeSeq-window_length_rbp+1):
        string = inputSeq[i:i+window_length_rbp]
#         print string
        score =cal_window_score(window_length_rbp, PWM, string,rbp)
#         print score/max_score*100
        stringScore[str(i)+'_'+string] = (score/max_score)*100
#         i+=1
#     print inputSeq
    return stringScore

def findScoreForRBP(rbp, inputSeq, PWM, threshold):
    window_length_rbp = find_len_of_rbp(rbp,PWM)
#     print 'len', window_length_rbp
    max_score = find_max_score_for_rbp(rbp,PWM,window_length_rbp)
#     print 'max', max_score
    stringScore = stringWindowScore(inputSeq,window_length_rbp, max_score, PWM,rbp)
#     print stringScore
    count = findCountAboveThreshold(stringScore, threshold)
    return count


def findCountAboveThreshold(stringWindowScore, threshold):
    thresholdList= []
    for key in stringWindowScore:
        if stringWindowScore[key]>=threshold:
#             print 'key', key, stringWindowScore[key]
            thresholdList.append(key)
    return len(thresholdList)

def getRBPfeatureCount(PWM, inputSeq, threshold):
    rbpCountList=[]
    for rbp in PWM:
        rbpCountList.append(findScoreForRBP(rbp, inputSeq, PWM, threshold))
    return rbpCountList
   
# def createFeatureVectorSpace(RNASeq, yLabel, PWM, threshold,rbpList):
#     print rbpList
#     values = RNASeq.values()
#     dfRBPs = pd.DataFrame(index=values,columns=rbpList)
#     #for each input sequence, features corresponding to each rbp
#     for i in range(len(values)):
#         inputSeq = values[i]
#         print inputSeq
#         rbpCountListforInputSeq=[]
#         for rbp in PWM:
#         #     print "pwm[rbp]", PWM[rbp]
#             print rbp
#             rbpCountListforInputSeq.append(findScoreForRBP(rbp, inputSeq, PWM, threshold))
#         print rbpCountListforInputSeq
#         dfRBPs.loc[inputSeq]=rbpCountListforInputSeq
#     dfRBPs['label']=yLabel
#     return dfRBPs
