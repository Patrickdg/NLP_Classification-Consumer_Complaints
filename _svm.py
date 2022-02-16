import model_funcs as mf

import os
import pandas as pd
import numpy as np
import nltk
import pickle

import sklearn
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV

if __name__ == "__main__":
    OVERWRITE_MODEL = True
    MODEL = 'svm'
    grid_params = {
        'SVC__C': [0.1, 1.0, 10.0, 20.0], 
        'SVC__kernel': ['rbf', 'linear', 'sigmoid'],
        'SVC__decision_function_shape': ['ovo', 'ovr']
    }
    # grid_params = {
    #     'SVC__C': [0.1], 
    #     'SVC__kernel': ['rbf'],
    #     'SVC__decision_function_shape': ['ovo']
    # }
    #Data - raw
    train_x, test_x, train_y, test_y, label_map  = mf.get_train_test()
    #Model
    pipeline = Pipeline(steps = [
        ("Normalize", Normalizer()),
        ("SVC", SVC())])
    pipeline = GridSearchCV(pipeline, grid_params, cv = 3)
    data = [pipeline, train_y, test_y, label_map]
    model = mf.run_all(data, MODEL, OVERWRITE_MODEL)