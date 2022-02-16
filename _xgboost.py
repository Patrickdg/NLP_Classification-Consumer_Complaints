import model_funcs as mf

import os
import pandas as pd
import numpy as np
import pickle
import gensim

import xgboost as xgb
from xgboost import XGBClassifier
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, f1_score

if __name__ == "__main__": 
    OVERWRITE_MODEL = True
    MODEL = 'xgb'
    grid_params = {
        'XGB__eta': [0.01, 0.5, 1.0],
        'XGB__gamma': [0.0, 5, 10],
        'XGB__max_depth': [3, 6]
        }
    #Data - raw
    train_x, test_x, train_y, test_y, label_map  = mf.get_train_test()
    #Model
    pipeline = Pipeline([
        ("XGB", XGBClassifier(use_label_encoder = False, eval_metric = 'mlogloss'))])
    pipeline = GridSearchCV(pipeline, grid_params, cv = 3)
    data = [pipeline, train_y, test_y, label_map]
    model = mf.run_all(data, MODEL, OVERWRITE_MODEL)