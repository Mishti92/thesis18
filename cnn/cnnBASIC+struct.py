
# coding: utf-8

# In[1]:


#Import files
import numpy as np
import sys
sys.path.append('../')

from sklearn import metrics
from sklearn.metrics import roc_auc_score

from Bio import SeqIO
import csv
import pandas as pd
import random
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle
import pdb
import gzip
import time

# ENcoding
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from keras.models import Sequential, model_from_config
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input, merge, LSTM, Merge
from keras.models import Model # basic class for specifying and training a neural network
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, RMSprop, Adadelta, Adagrad, Adam
from keras.layers import normalization
from keras.layers.convolutional import Conv1D,Convolution2D, MaxPooling2D
from keras.layers import LSTM, Bidirectional, concatenate
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution2D, MaxPooling2D,Convolution1D, MaxPooling1D
from keras import regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.utils.vis_utils import plot_model
from keras.callbacks import History 
from keras.callbacks import TensorBoard

# plfold
import os
import numpy as np
import sys
sys.path.append('../')
from matplotlib import pyplot as plt
import scipy.stats as stats
import csv
import pandas as pd
import random
from itertools import islice
import pickle
import itertools
import random
from os import listdir
from  __builtin__ import any
 

def read_seq(seq_file,plfold):
    seq_list = []
    seq = ''
    with open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
                if len(seq):
                    seq_array = get_RNA_seq_concolutional_array(seq, name, plfold)
                    seq_list.append(seq_array)                    
                seq = ''
            else:
                seq = seq + line[:-1]
        if len(seq):
            seq_array = get_RNA_seq_concolutional_array(seq, name, plfold)
            seq_list.append(seq_array) 
    
    return np.array(seq_list)

def load_label_seq(seq_file):
    label_list = []
    seq = ''
    with open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
                posi_label = name.split(';')[-1]
                label = posi_label.split(':')[-1]
                label_list.append(int(label))
    return np.array(label_list)

def get_RNA_seq_concolutional_array(seq, key, plfold, motif_len = 10):
    seq = seq.replace('U', 'T')
#     alpha = 'acgu'
    alpha = 'ACGT'
#     print seq
    half_len = motif_len/2
    row = (len(seq) + half_len *2 )
    new_array = np.zeros((row, 5))

#     First and last half_len values set to 0.25

    for i in range(half_len):
        new_array[i] = np.array([0.25]*5)
    
    for i in range(row-half_len, row):
        new_array[i] = np.array([0.25]*5)    

        
    for i, val in enumerate(seq):
#         print i,val
        i = i + half_len
#         print i,val
#         if val not in 'acgu':
        if val not in 'ACGT':
            new_array[i] = np.array([0.25]*5)
            continue
        #if val == 'N' or i < motif_len or i > len(seq) - motif_len:
        #    new_array[i] = np.array([0.25]*4)
        #else:
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
        #data[key] = new_array
        
#         using the key, find the file in the given protein directory
#  Extract probabilities, then take sum of squares of probabilities and put in the array

    for name in plfold:
        if name == key:
            for i, val in enumerate(seq):
#                 i = i + half_len
                new_array[i + half_len][4] = plfold[name][i]
        
    return new_array

def load_data_file(inputfile, plfold, seq = True, onlytest = False):
    """
        Load data matrices from the specified folder.
    """
    
    path = os.path.dirname(inputfile)
    data = dict()
    if seq: 
        tmp = []
        tmp.append(read_seq(inputfile, plfold))
#         seq_onehot, structure = read_structure(inputfile, path)
#         tmp.append(seq_onehot)
        data["seq"] = tmp
#         data["structure"] = structure
    if onlytest:
        data["Y"] = []
    else:
        data["Y"] = load_label_seq(inputfile)
        
    return data


def split_training_validation(classes, validation_size, shuffle = True):
    """split sampels based on balnace classes"""
    num_samples=len(classes)
    classes=np.array(classes)
    classes_unique=np.unique(classes)
    num_classes=len(classes_unique)
    indices=np.arange(num_samples)
    #indices_folds=np.zeros([num_samples],dtype=int)
    training_indice = []
    training_label = []
    validation_indice = []
    validation_label = []
    for cl in classes_unique:
        indices_cl=indices[classes==cl]
        num_samples_cl=len(indices_cl)

        # split this class into k parts
        if shuffle:
            random.shuffle(indices_cl) # in-place shuffle
        
        # module and residual
        num_samples_each_split=int(num_samples_cl*validation_size)
        res=num_samples_cl - num_samples_each_split
        
        training_indice = training_indice + [val for val in indices_cl[num_samples_each_split:]]
        training_label = training_label + [cl] * res
        
        validation_indice = validation_indice + [val for val in indices_cl[:num_samples_each_split]]
        validation_label = validation_label + [cl]*num_samples_each_split

    training_index = np.arange(len(training_label))
    random.shuffle(training_index)
    training_indice = np.array(training_indice)[training_index]
    training_label = np.array(training_label)[training_index]
    
    validation_index = np.arange(len(validation_label))
    random.shuffle(validation_index)
    validation_indice = np.array(validation_indice)[validation_index]
    validation_label = np.array(validation_label)[validation_index]    
    
            
    return training_indice, training_label, validation_indice, validation_label    
 

def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder

def calculate_performace(test_num, pred_y,  labels):
    tp =0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] ==1:
            if labels[index] == pred_y[index]:
                tp = tp +1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn +1
            else:
                fp = fp + 1               
            
    acc = float(tp + tn)/test_num
    precision = float(tp)/(tp+ fp)
    sensitivity = float(tp)/ (tp+fn)
    specificity = float(tn)/(tn + fp)
    MCC = float(tp*tn-fp*fn)/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    return acc, precision, sensitivity, specificity, MCC 


def list_files(directory, extension):
    return (f for f in listdir(directory) if f.endswith('.' + extension))


def list_dir(path):
    proteinList=[]
    files = os.listdir(path)
    for name in files:
        proteinList.append(name)
    return proteinList


def accuracy(test_x, test_y, model):
    result = model.predict(test_x)
    predicted_class = np.argmax(result, axis=1)
    true_class = np.argmax(test_y, axis=1)
    num_correct = np.sum(predicted_class == true_class) 
    accuracy = float(num_correct)/result.shape[0]
    return (accuracy * 100)

def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()


# In[2]:


def run_network_new(name, training, y, validation, val_y, batch_size=50, nb_epoch=30):
    print 'configure cnn network'
    
    rows=111
    cols=5
    
    model = Sequential()
    
    input_shape = Input(shape=(rows,cols))
    tower_1 = Conv1D(16, 10, padding='valid', activation='relu')(input_shape) 
    tower_temp = Activation('relu')(tower_1)
    pool = MaxPooling1D(pool_size=3, padding='valid')(tower_temp)
    flat = Flatten()(pool)
    out = Dense(50, activation='relu')(flat)
    out = Dense(2, activation='sigmoid')(out)
    model = Model(input_shape, out)
 

    history = History()
    
    model.compile(loss=
#                   'kullback_leibler_divergence'
                  'categorical_crossentropy'
                  , optimizer=RMSprop(lr=0.001),metrics=['accuracy'])
    
    model.summary()
    
    plot_model(model, to_file='architecture/%s.png'%(name))
    
    #tensorboard = TensorBoard(log_dir='Graph', histogram_freq=0, write_graph=True, write_images=True)
    #pdb.set_trace()
    print 'model training'

    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=0)

    history = model.fit(training, y, batch_size=batch_size, epochs=nb_epoch, verbose=0, validation_data=(validation, val_y)
              , callbacks=[history
                           ,earlystopper
                           #,tensorboard
                          ])

    return model, history 


# In[3]:


dir_path = "/mnt/mirror/data/mimtiy/lab/encode/data/proteins/"
proteins = list_dir(dir_path)

train_path="/30000/training_sample_0/"
test_path ="/30000/test_sample_0/"


# In[4]:


print "Starting with the code (With Structural Features)" 

#for protein in proteins:
for i in range(len(proteins)):
#i=1
    protein = proteins[i]
    print "******************************************************"
    print "#", i ," - Protein: ", protein
    print "******************************************************"

    train_file_path = dir_path + protein + train_path + 'sequences.fa'
    test_file_path = dir_path + protein + test_path + 'sequences.fa'

    with open('pickle/train_plfold_%s.pickle'%(protein), 'rb') as handle:
        train_plfold = pickle.load(handle)
    with open('pickle/test_plfold_%s.pickle'%(protein), 'rb') as handle:
        test_plfold = pickle.load(handle)

    training_data = load_data_file(train_file_path, train_plfold)
    test_data = load_data_file(test_file_path, test_plfold)

        #Saving to Label and Seq 
    train_Y = training_data["Y"]
    test_label = test_data["Y"]
    seq_data = training_data["seq"][0]
    seq_test = test_data["seq"][0]


    print "Training Examples: ", train_Y.shape
    print "Test Examples: ", test_label.shape
    print "Train Set Dimensions: ", seq_data.shape
    print "Test Set Dimensions: ", seq_test.shape


    # In[5]:


    # Train Test Split
    training_indice, training_label, validation_indice, validation_label = split_training_validation(train_Y,0.2, False)
    seq_train = seq_data[training_indice]
    seq_validation = seq_data[validation_indice] 

    print "Training Set Shape: ", seq_train.shape, training_label.shape
    print "Validation Set Shape: ", seq_validation.shape, validation_label.shape
    print "Test Set Shape: ", seq_test.shape, test_label.shape

    # In[7]:
    name = 'cnnBASIC+struct'
    batch_size= 50 
    nb_epoch = 30
    seq_data = []

    y, encoder = preprocess_labels(training_label)
    val_y, encoder = preprocess_labels(validation_label, encoder = encoder) 
    test_y, encoder = preprocess_labels(test_label)

    training_data.clear()

    start = time.time()
    model, history = run_network_new(name, seq_train, y, validation = seq_validation,val_y = val_y, batch_size=batch_size, nb_epoch = nb_epoch)
    end = time.time()

    print "Model took %0.2f seconds to train"%(end - start)


    # In[8]:


    # print(history.history.keys())
    # print(history.history['val_acc'])
    # plot_model_history(history).savefig("figure/%s_%s.png"%(protein,name))


    # Evaluate the trained model on the test set!

    print "Accuracy of Test Set:", model.evaluate(seq_test,test_y, verbose=0)[1]
    predictions = model.predict(seq_test, verbose=0)
     # print predictions
    auc = roc_auc_score(test_y, predictions)
    print "Test AUC: ", auc
    print model.evaluate(seq_test,test_y, verbose=0)[1], auc

    print "******************************************************"
    print "#", i ," - Protein: ", protein
    print "******************************************************"
