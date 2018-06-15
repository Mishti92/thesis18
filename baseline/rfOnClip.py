import numpy as np
import sys
sys.path.append('../')
sys.path.append('../utils/')
from Bio import SeqIO
from matplotlib import pyplot as plt
import csv
import pandas as pd
import readMethods
import methods
import random
import pickle
import os
import sklearn
import sklearn.ensemble
from sklearn.model_selection import train_test_split
from sklearn import svm
from collections import Iterable
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics
from sklearn.metrics import roc_auc_score

from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn import linear_model
from sklearn.cross_validation import KFold

path = "/mnt/mirror/data/mimtiy/lab/encode/"
with open(path + 'files/pickle/PWM.pickle', 'rb') as handle:
        PWM = pickle.load(handle)
        
with open(path + 'files/pickle/threshold.pickle', 'rb') as handle:
        thresholds = pickle.load(handle)

def flatten(lis):
     for item in lis:
            if isinstance(item, Iterable) and not isinstance(item, basestring):
                for x in flatten(item):
                    yield x
            else:
                yield item
                
def list_dir(path):
    proteinList=[]
    files = os.listdir(path)
    for name in files:
        proteinList.append(name)
    return proteinList

def read_seq(seq_file):
    seq_list = []
    label_list=[]
    seq = ''
    with open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
                posi_label = name.split(';')[-1]
                label = posi_label.split(':')[-1]
#                 print name
                if len(seq):
                    if 'N' not in seq:
                        seq_list.append(seq.replace('T','U').lower())  
                        label_list.append(int(label))
                seq = ''
            else:
                seq = seq + line[:-1]
        if len(seq):
            if 'N' not in seq:
                seq_list.append(seq.replace('T','U').lower())
                label_list.append(int(label))
    return seq_list, np.array(label_list)

def load_data_file(inputfile):
    path = os.path.dirname(inputfile)
    data = dict()
    seq, label = read_seq(inputfile)
    return seq, label

def createFeatureVectors(data, thresholdSet):
    features=[]
    for i in range(len(data)):
        inputSeq= data[i]
        rbpCountListforInputSeq=[]
        for rbp in PWM:
            if thresholdSet == "0":
                threshold = 80
            elif thresholdSet == "1":
                threshold = thresholds[rbp]*100
            temp = methods.findScoreForRBP(rbp, inputSeq, PWM, threshold)
            rbpCountListforInputSeq.append(temp)
        features.append(rbpCountListforInputSeq)
    return features


rbpList =[]
for rbp in PWM:
    rbpList.append(rbp)

featureNames = rbpList
thresholdSet = '0'

    
dir_path = "/mnt/mirror/data/mimtiy/lab/encode/data/proteins/"
proteins = list_dir(dir_path)
for i in range(len(proteins)):
    protein = proteins[i]
    print "******************************************************"
    print "#", i ," - Protein: ", protein
    print "******************************************************"
    train_path = dir_path + protein + "/30000/training_sample_0/sequences.fa"
    test_path = dir_path + protein + "/30000/test_sample_0/sequences.fa"
    
    train_data, train_label=load_data_file(train_path)
    test_data, test_label=load_data_file(test_path)
    
    print "Reading Features and Labels"
    #train_features = createFeatureVectors(train_data, thresholdSet)
    #test_features = createFeatureVectors(test_data, thresholdSet)
    
    #with open('../pickle/train_features_%s.pickle'%(protein), 'wb') as handle:
    #    pickle.dump(train_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #with open('../pickle/test_features_%s.pickle'%(protein), 'wb') as handle:
    #    pickle.dump(test_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
     
        
    with open('../pickle/train_features_%s.pickle'%(protein), 'rb') as handle:
        train_features = pickle.load(handle)
    
    with open('../pickle/test_features_%s.pickle'%(protein), 'rb') as handle:
        test_features = pickle.load(handle)
        
        
    print "No. of Training features: ",len(train_features[0])
    print "Total no. of training instances: ", len(train_features)
    X = np.array(train_features)
    y = train_label
    # print X.shape
    # print y.shape

    print "No. of Test features: ",len(test_features[0])
    print "Total no. of test instances: ", len(test_features)
    X_test = np.array(test_features)
    y_test = test_label

    print "Done..."
    
    print "Random Forest Classifier - 5 fold cross validation"
    lw = 1.2
    kfold = 5

    random_state = np.random.RandomState(0)
    kf = KFold(n=y.shape[0], n_folds=kfold)
    classifier =  sklearn.ensemble.RandomForestClassifier(n_estimators=100
                                                      ,random_state=random_state
                                                         )
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    for i, (train, valid) in enumerate(kf):
        model = classifier.fit(X[train], y[train])
        y_score = model.predict_proba(X[valid])
        y_pred = model.predict(X[valid])
        target_names = ['0', '1']
    #     print classification_report(y[valid], y_pred,target_names=target_names)

    #         # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[valid], y_score[:, 1])
    #     aucValue = roc_auc_score(y[valid], y_pred)
    #     print "Validation AUC: ", aucValue
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)

        mean_tpr /= kfold
        mean_tpr[-1] = 1.0

        mean_auc = auc(mean_fpr, mean_tpr)

        print 'Mean Validation AUC', mean_auc

    print "Test Set Evaluation"
    y_test_score = model.predict_proba(X_test)    
    y_test_pred = model.predict(X_test)
    target_names = ['0', '1']
    aucTestValue = roc_auc_score(y_test,y_test_pred)
    num_correct = np.sum(y_test_pred == y_test) 
    accuracy = float(num_correct)/y_test_pred.shape[0]
    fpr, tpr, thresholds = roc_curve(y_test, y_test_score[:, 1])
    roc_test_auc = auc(fpr, tpr)

    print "Test AUC: ", aucTestValue
    print "Accuracy of Test Set       Test Set ROC"
    print (accuracy * 100), "            ", roc_test_auc
    print "******************************************************"
    print "#", i ," - Protein: ", protein
    print "******************************************************"


