import numpy as np
from sklearn.metrics import average_precision_score
import tensorflow as tf
from tensorflow.keras.models import  Model
import numpy as np
import datetime
import random
import time
import json
from itertools import groupby
import logging
import logging.config
import pickle
import pandas as pd
import importlib

def avg_precision(y_true, y_pred):
    if sum(y_true) == 0:
        return 1
    avp = average_precision_score(y_true, y_pred)
    if np.isnan(avp):
        return 0
    else:
        return avp

"""
We have change the definition because in QA we do not have order, we just 
need to retrieve related answer, so if the correct answer is retrieved it will have an
score of 1 other wise
"""
def reciprocal_rank(y_true, y_pred):
    return 0

class MAPCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, val_data, filepath, predict_fc, min_delta=0, patience=50, verbose=1, save_best_only=True, save_weights_only=False, period=1):
        super(MAPCallback, self).__init__()
        self.predict_fc = predict_fc
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
        self.val_data = val_data
        
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
                            print('Epoch %05d: %s did not improve' %
                                  (epoch, ' MAP '))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)

    def calculate_map_mrr(self):
        avg_p = []
        rr_list = []
        num_groups = 0
        count = 0
        for key, group in groupby(sorted(self.val_data,key=lambda x: x['id']), lambda x: x['id']):
            count += 1
            y_true = []
            y_pred = []
            x_samples = []
            for answer in group:
                x_samples.append(answer['representation'])
                y_true.append(answer['label'])
            if len(y_true) > 1:
                y_pred = self.predict_fc(x_samples)
                avg_p_score = avg_precision( y_true, y_pred )
                avg_p.append( avg_p_score )
                rr_score = reciprocal_rank( y_true, y_pred )
                rr_list.append( rr_score )
                num_groups += 1
        cmap_score = sum(avg_p)/num_groups
        print("groups #",num_groups)
        print("validation map #",cmap_score)
        cmrr_score = sum(rr_list)/num_groups
        print("validation mrr #",cmrr_score)
        self.map_score.append(cmap_score)
        self.mrr_score.append(cmrr_score)
        return cmap_score, cmrr_score