# LIBRARIES
import model_funcs as mf

import os
import math
import random
import pandas as pd  
import numpy as np
import nltk
import pickle   
from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split

#OVERWRITE CONTROLS
OVERWRITE_TRAIN_TEST = False
OVERWRITE_EMBEDDINGS = False
OVERWRITE_CUSTOM_EMBEDDINGS = False
OVERWRITE_RESULTS_DF = False

#PARAMETERS
DATA_PATH = r'C:\Users\deguz\OneDrive\PET_PROJECTS\NLP_Classification-Consumer_Complaints\_data\complaints_processed.csv'
SPLIT_PATH = r'C:\Users\deguz\OneDrive\PET_PROJECTS\NLP_Classification-Consumer_Complaints\_data\_processed'
RESULTS_PATH = 'results.csv'

SAMPLE_RATE = 0.05
TEST_SIZE = 0.2 # % of data to use as test set

INPUT_COL = 'narrative'
TARGET_COL = 'product'

WORD2VEC_PATH = 'embeddings\GoogleNews-vectors-negative300.bin.gz'
GLOVE_PATH = 'embeddings\glove.6B\glove.6B.100d.txt'
FASTTEXT_PATH = 'embeddings\wiki-news-300d-1M.vec'
CUSTOM_PATH = 'embeddings\custom_embeddings.model'

#FUNCTIONS 
## Reading
def compile_data(train_path, input_col, target_col, frac = 0.10): 
    train = pd.read_csv(train_path)
    train.dropna(inplace = True); train.reset_index(inplace = True)
    train = train.sample(frac = frac)

    train_x = list(train[input_col])
    train_y = train[target_col]
    label_map = {label: code for label, code in zip(train_y, train_y.astype('category').cat.codes)}
    train_y = train[target_col].astype('category').cat.codes

    label_map = dict(sorted(label_map.items(), key = lambda item: item[1]))
    print("==== DATA COMPILED")
    return train_x, train_y, label_map

## Tokenization
def tokenize(samples): 
    tokens = []
    for sample in samples: 
        token_list = nltk.tokenize.RegexpTokenizer("['\w]+").tokenize(sample)
        tokens.append(token_list)
    print("==== DATA TOKENIZED")
    return tokens

## Undersampling
def undersampling(train_x, train_y, label_map, target_class, factor):
    target_class = label_map[target_class]
    sampled_x = []
    sampled_y = []
    for c_name, c in label_map.items():
        c_mask = [y == c for y in train_y]
        samples_x = [x for m, x in zip(c_mask, train_x) if m]
        samples_y = [y for m, y in zip(c_mask, train_y) if m]
    
        if c == target_class: 
            num_samples = math.ceil(len(samples_x)/factor)
            samples_x, samples_y = zip(*random.sample(list(zip(samples_x, samples_y)), num_samples))

        sampled_x.extend(samples_x)
        sampled_y.extend(samples_y)
        print(f"   {len(samples_y)} samples selected of {c_name}")

    return sampled_x, sampled_y

## Train/test split
def split_data(train_x, train_y, label_map, test_size, path, overwrite_flag): 
    if overwrite_flag: 
        train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size = test_size)

        names = ['train_x', 'test_x', 'train_y', 'test_y', 'label_map']
        vars = [train_x, test_x, train_y, test_y, label_map]

        print(f"==== DATA SPLIT & SAVED")
        for n, v in zip(names, vars): 
            with open(f'{path}\{n}.pkl', 'wb') as f: 
                pickle.dump(v, f)
    else: 
        train_x, test_x, train_y, test_y, label_map  = mf.get_train_test()
    return train_x, test_x, train_y, test_y, label_map

def build_custom_embeddings(train_x, overwrite_flag):
    if overwrite_flag: 
        vec_model = Word2Vec(sentences = train_x, vector_size = 300, min_count = 2, workers = 4)
        vec_model.save(CUSTOM_PATH)
        print(f"==== CUSTOM EMBEDDINGS SAVED")

def build_embeddings(overwrite_flag):
    if overwrite_flag: 
        emb_dict = mf.embedding_funcs
        for emb_name, emb_func in emb_dict.items(): 
            train_path = f'embedding_vecs/{emb_name}/train_x_vecs.pkl'
            test_path = f'embedding_vecs/{emb_name}/test_x_vecs.pkl'
            if not os.path.exists(train_path): 
                try: 
                    os.mkdir(f'embedding_vecs/{emb_name}')
                except: 
                    pass
                emb_func = emb_dict[emb_name]
                embeddings_index = emb_func()
                train_x_vecs = mf.create_doc_vecs(train_x, embeddings_index)
                test_x_vecs = mf.create_doc_vecs(test_x, embeddings_index)
                for p, vec in zip([train_path, test_path], [train_x_vecs, test_x_vecs]): 
                    with open(p, 'wb') as f: 
                        pickle.dump(vec, f)

## Results summary df
def build_results_df(path, overwrite_flag): 
    if overwrite_flag or not os.path.exists(path):
        results = pd.DataFrame(columns = ['model_name', 'accuracy', 'weighted_f1'])
        print(f"==== RESULTS DF SAVED")
        results.to_csv(path, index=False)

if __name__ == "__main__": 
    train_x, train_y, label_map  = compile_data(DATA_PATH, INPUT_COL, TARGET_COL, frac = SAMPLE_RATE)
    train_x = tokenize(train_x)
    train_x, train_y = undersampling(train_x, train_y, label_map, 'credit_reporting', 4) #Imbalanced data, 'credit_reporting' class = 56% of samples 
    train_x, test_x, train_y, test_y, label_map = split_data(train_x, train_y, label_map, TEST_SIZE, SPLIT_PATH, OVERWRITE_TRAIN_TEST)
    build_custom_embeddings(train_x, OVERWRITE_CUSTOM_EMBEDDINGS)
    build_embeddings(OVERWRITE_EMBEDDINGS)
    build_results_df(RESULTS_PATH, OVERWRITE_RESULTS_DF)