import sys
import threading
from os import listdir
from os.path import isfile, join
import time
import os
import pandas as pd
import json
import logging
from tqdm import tqdm
import p_tqdm
from itertools import groupby
import sys
import pickle
import numpy as np
from scipy import spatial
import re
from random import shuffle
import random
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback

def precision(y_true, y_pred):
    if sum(y_true) == 0:
        return 1
    return float(sum(np.multiply(y_true, y_pred))/sum(y_true))

def avg_precision(y_true, y_pred):
    score = 0
    for i in range(len(y_true)):
        print(precision(y_true[0:i+1],y_pred[0:i+1]))
        score += precision(y_true[0:i+1],y_pred[0:i+1])
    return score/len(y_true)

"""
https://en.wikipedia.org/wiki/Mean_reciprocal_rank
"""
def reciprocal_rank(y_true, y_pred):
    zipped = list(zip(y_true, y_pred))
    zipped.sort(key=lambda x:x[1],reverse=True)
    count_r = 1.0
    rr_score = 0.0
    for y_t,y_p in zipped:
        if(y_t!=1):
            count_r += 1
        else:
            rr_score = 1.0/count_r
            break
    if count_r-1==len(y_true):
        rr_score = 0.0
    return rr_score

class MAPCallback(Callback):
    
    def __init__(self, validation_data, max_words, proc_funct, filepath, min_delta=0, patience=50, verbose=1, save_best_only=True, save_weights_only=True, period=1):
        super(MAPCallback, self).__init__()
        self.val_ds = validation_data
        self.max_words = max_words
        self.proc_fuction = proc_funct
        self.map_score = []
        self.mrr_score = []
        self.min_delta = min_delta
        #maximize the map
        self.monitor_op = np.greater
        self.patience = patience
        self.period = period
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.epochs_since_last_save = 0
        self.filepath = filepath
        self.min_delta *= -1
        self.stopped_epoch = 0
        self.test_data = []
        x, y = buildCosineSimMatrix(validation_data, max_terms=max_terms)
        x = list(x)
        y = list(y)
        for i, qa in enumerate(validation_data):
            self.test_data.append( (x[i], y[i], qa.qi) )
    
    def on_train_begin(self, logs=None):
        self.wait = 0  # Allow instances to be re-used
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs={}):
        current_map, current_mrr = self.calculate_map_mrr()
        self.save_model(epoch, logs, current_map)
        logging.info("MAP evaluation - epoch: {:d} - score: {:.6f}".format(epoch, current_map))
        logging.info("MRR evaluation - epoch: {:d} - score: {:.6f}".format(epoch, current_mrr))
        if current_map is None:
            warnings.warn('MAP Early stopping requires %s available!' %
                          (self.monitor), RuntimeWarning)

        if self.monitor_op(current_map - self.min_delta, self.best):
            self.best = current_map
            self.wait = 0
        else:
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
            self.wait += 1

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
        #    print('Epoch %05d: MAP early stopping' % (self.stopped_epoch))
            logging.info("MAP early stopping Epoch {:d} evaluation - MAP: {:.6f} ".format(self.stopped_epoch,self.best))
    
    def save_model(self, epoch, logs, cmap):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch, **logs)
            if self.save_best_only:
                current = cmap
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (' MAP '), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch, ' MAP ', self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve current( %0.5f), best( %0.5f)' %
                                  (epoch, ' MAP ', current, self.best))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)

    def calculate_map_mrr(self):
        y_true = []
        y_pred = []
        avg_p = []
        rr_list = []
        num_groups = 0
        for key, group in groupby(self.test_data, lambda x: x[2]):
            samples = []
            y_true = []
            is_any_true = False
            for q_g in group:
                samples.append(q_g[0])
                y_true.append(q_g[1])
                if int(q_g[1]) == 1:
                    is_any_true = True
            #Just take the 1/3 of the dataset randomly
            if random.uniform(0, 10) > 8:
                is_any_true = False
            if is_any_true:
                y_pred = self.model.predict(np.array(samples))
                avg_p_score = avg_precision( y_true, y_pred )
                avg_p.append( avg_p_score )
                rr_score = reciprocal_rank( y_true, y_pred )
                rr_list.append( rr_score )
                num_groups += 1
        cmap_score = sum(avg_p)/num_groups
        cmrr_score = sum(rr_list)/num_groups
        self.map_score.append(cmap_score)
        self.mrr_score.append(cmrr_score)
        #print 'map = ', cmap_score, ', mrr = ', cmrr_score
        return cmap_score, cmrr_score