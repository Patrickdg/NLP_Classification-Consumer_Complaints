# LIBRARIES
import model_funcs as mf

import os
import re
import math
import random
import pandas as pd  
import numpy as np
import nltk
from nltk.corpus import wordnet
from nltk import RegexpTokenizer
import pickle   
from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split

#OVERWRITE CONTROLS
OVERWRITE_TRAIN_TEST = False
OVERWRITE_RESULTS_DF = False
OVERWRITE_PARAMS_DF = False
OVERWRITE_EMB_FLAGS = {
    'word2vec': False, 
    'glove': False, 
    'fasttext': False,
    'custom': False
}

#PARAMETERS - Data
DATA_PATH = r'C:\Users\deguz\OneDrive\PET_PROJECTS\NLP_Classification-Consumer_Complaints\_data\complaints_processed.csv'
SPLIT_PATH = r'C:\Users\deguz\OneDrive\PET_PROJECTS\NLP_Classification-Consumer_Complaints\_data\_processed'
RESULTS_PATH = 'results/'

SAMPLE_RATE = 0.075
TEST_SIZE = 0.1 # % of data to use as test set
AUG_P = 0.0 # % of words to augment(syonyms) per sample
AUG_RETURN_P = 0.0 # % to increase training set in augmentation

INPUT_COL = 'narrative'
TARGET_COL = 'product'

#PARAMETERS - Embeddings
WORD2VEC_PATH = 'embeddings\GoogleNews-vectors-negative300.bin.gz'
GLOVE_PATH = 'embeddings\glove.6B\glove.6B.100d.txt'
FASTTEXT_PATH = 'embeddings\wiki-news-300d-1M.vec'
CUSTOM_PATH = 'embeddings\custom_embeddings.model'

CUSTOM_VECTOR_SIZE = 300
CUSTOM_MIN_COUNT = 2

#FUNCTIONS 
## Reading
def compile_data(train_path, input_col, target_col, frac): 
    """
    Process: 
    - drops null values 
    - samples for 'frac' % of total dataset
    - generates label_map for coding of classes/labels
    """
    train = pd.read_csv(train_path)
    train.dropna(inplace = True); train.reset_index(inplace = True)
    train = train.sample(frac = frac)

    train_x = list(train[input_col])
    train_y = train[target_col]
    label_map = {label: code for label, code in zip(train_y, train_y.astype('category').cat.codes)}
    train_y = train[target_col].astype('category').cat.codes

    label_map = dict(sorted(label_map.items(), key = lambda item: item[1]))
    print(f"==== DATA COMPILED, Num. samples = {len(train_x)}")
    return train_x, train_y, label_map

## Tokenization
def tokenize(samples): 
    tokens = []
    for sample in samples: 
        sample = re.sub(r'(x{2,})', '', sample)
        token_list = nltk.tokenize.RegexpTokenizer("['\w]+").tokenize(sample)
        token_list = [t for t in token_list if len(t) > 2]
        tokens.append(token_list)
    print("==== DATA TOKENIZED")
    return tokens

## Undersampling for a target_class (overrepresentation)
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
    print(f"{len(sampled_x)} total samples remaining")
    return sampled_x, sampled_y

#DATA AUGMENTATION: Synonyms
def sentence_aug_synonym(data_x, data_y, aug_p = 0.5, return_p = 0.5):
    aug_data_x = []
    aug_data_y = []
    for x, y in zip(data_x, data_y): 
        aug_x = []
        if random.random() <= return_p:
            for w in train_x[0]:
                if random.random() <= aug_p: 
                    try: 
                        syn = random.choice(wordnet.synsets(w))
                        syn_w = random.choice([l.name() for l in syn.lemmas()])
                        aug_x.append(syn_w)
                        # print(w, "---->", syn_w)
                    except: 
                        aug_x.append(w)
            aug_data_x.append(aug_x)
            aug_data_y.append(y)
    data_x.extend(aug_data_x)
    data_y.extend(aug_data_y)
    return data_x, data_y

## Train/test split
def split_data(train_x, train_y, label_map, test_size, path, overwrite_flag): 
    if overwrite_flag: 
        train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size = test_size)
        # train_x, train_y = sentence_aug_synonym(train_x, train_y, AUG_P, AUG_RETURN_P)

        names = ['train_x', 'test_x', 'train_y', 'test_y', 'label_map']
        vars = [train_x, test_x, train_y, test_y, label_map]

        print(f"==== DATA SPLIT & SAVED, Total samples = {len(train_x)}")
        for n, v in zip(names, vars): 
            with open(f'{path}\{n}.pkl', 'wb') as f: 
                pickle.dump(v, f)
    else: 
        train_x, test_x, train_y, test_y, label_map  = mf.get_train_test()
    return train_x, test_x, train_y, test_y, label_map

#Embeddings
def build_custom_embeddings(train_x):
    vec_model = Word2Vec(sentences = train_x, 
                        vector_size = CUSTOM_VECTOR_SIZE, 
                        min_count = CUSTOM_MIN_COUNT, 
                        workers = 4)
    vec_model.save(CUSTOM_PATH)
    print(f"==== CUSTOM EMBEDDINGS SAVED")

def build_embeddings(overwrite_emb_flags):
    emb_dict = mf.embedding_funcs
    for emb_name, emb_func in emb_dict.items(): 
        if overwrite_emb_flags[emb_name]: 
            if emb_name == 'custom': #Special case: Build word2vec model first
                build_custom_embeddings(train_x)
            emb_path = f'embedding_vecs/{emb_name}'
            train_path = f'embedding_vecs/{emb_name}/train_x_vecs.pkl'
            test_path = f'embedding_vecs/{emb_name}/test_x_vecs.pkl'
            if not os.path.exists(emb_path): 
                try: 
                    os.mkdir(emb_path)
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
def build_results_dfs(path, overwrite_flag, overwrite_params): 
    if not os.path.exists(path): 
        os.mkdir(path)

    if overwrite_flag:
        results = pd.DataFrame(columns = ['model_name', 'accuracy', 'f1'])
        print(f"==== RESULTS DF SAVED")
        results.to_csv(path + '/results.csv', index=False)

    if overwrite_params: 
        results = pd.DataFrame(columns = ['model_name', 'params'])
        print(f"==== PARAMS DF SAVED")
        results.to_csv(path + '/best_params.csv', index=False)

if __name__ == "__main__": 
    train_x, train_y, label_map  = compile_data(DATA_PATH, INPUT_COL, TARGET_COL, frac = SAMPLE_RATE)
    train_x = tokenize(train_x)
    train_x, train_y = undersampling(train_x, train_y, label_map, 'credit_reporting', factor = 4) #Imbalanced data, 'credit_reporting' class = 56% of samples 
    train_x, test_x, train_y, test_y, label_map = split_data(train_x, train_y, label_map, TEST_SIZE, SPLIT_PATH, OVERWRITE_TRAIN_TEST)
    build_embeddings(OVERWRITE_EMB_FLAGS)
    build_results_dfs(RESULTS_PATH, OVERWRITE_RESULTS_DF, OVERWRITE_PARAMS_DF)