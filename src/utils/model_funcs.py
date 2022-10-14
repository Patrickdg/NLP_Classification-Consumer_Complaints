import processing as p

import os
import numpy as np
import pandas as pd
import random
import pickle
import gensim

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt

"""GENERAL FUNCS======================================================"""
def get_train_test():
    names = ['train_x', 'test_x', 'train_y', 'test_y', 'label_map']
    variables = []
    for n in names: 
        with open(f'{p.SPLIT_PATH}/{n}.pkl', 'rb') as f: 
            variables.append(pickle.load(f))
    return variables

"""EMBEDDING-SPECIFIC FUNCS======================================================"""
#Generate doc vectors using word2vec/fasttext
def create_doc_vecs(docs, word_vecs): 
    try:
        word_vec_dim = word_vecs['test'].shape
    except:
        word_vec_dim = word_vecs[0].shape

    doc_vecs = []
    for doc in docs: 
        n_words = 0
        doc_vec = np.zeros(word_vec_dim)
        for word in doc:
            try:
                doc_vec = np.add(doc_vec, word_vecs[word])
                n_words += 1
            except: #Word not found in word2vec vocab
                continue
        doc_vec = doc_vec / n_words
        doc_vecs.append(doc_vec)
        
    return doc_vecs

#Retrieve build embeddings
def get_embedding_vecs(emb_name): 
    paths = ['train_x_vecs.pkl', 'test_x_vecs.pkl']
    vecs = []
    for p in paths:
        vec_path = os.path.join('embedding_vecs', emb_name, p) 
        with open(vec_path, 'rb') as f: 
            vecs.append(pickle.load(f))
    return vecs

#GLOVE
def build_glove_embeddings(): 
    embeddings_index = {}
    f = open(p.GLOVE_PATH, encoding = 'utf8')
    for line in f:
        values = line.split()
        word = values[0]
        embeddings_index[word] = np.asarray(values[1:], dtype='float32')
    f.close()

    return embeddings_index

#WORD2VEC
def build_word2vec_embeddings():
    word_vecs = gensim.models.KeyedVectors.load_word2vec_format(p.WORD2VEC_PATH, binary = True)
    embeddings_index = {}
    for word, idx in word_vecs.key_to_index.items():
        embeddings_index[word] = word_vecs[word]

    return embeddings_index    

#FASTTEXT
def build_fasttext_embeddings():
    embeddings_index = {}
    with open(p.FASTTEXT_PATH, encoding = 'utf8', newline='\n', errors='ignore') as f:
        for line in f: 
            values = line.split()
            word = values[0]
            embeddings_index[word] = np.asarray(values[1:], dtype='float32')
    f.close()

    return embeddings_index

#CUSTOM EMBEDDINGS
def build_custom_embeddings(): 
    word_vecs = gensim.models.Word2Vec.load(p.CUSTOM_PATH)
    embeddings_index = {}
    for word in word_vecs.wv.key_to_index:
        embeddings_index[word] = word_vecs.wv[word]

    return embeddings_index    

"""MODEL-SPECIFIC FUNCS======================================================"""
#BERT
def bert_processing(tokenizer, model_name, x, y, max_length): 
    tokenizer = tokenizer.from_pretrained(model_name)
    if isinstance(x[0], list): #samples are already tokenized: 
        x = list(map(' '.join, x)) 
        
    tokenized = tokenizer(
        x, 
        add_special_tokens=True, 
        max_length=max_length, 
        padding='max_length',
        return_attention_mask=True,
        truncation=True,
        return_tensors='tf')

    labels = np.array(pd.get_dummies(y).values)
    return [tokenized, labels]

def bert_train(model, train_data, test_data, epochs, batch, callbacks): 
    x_train, train_y = train_data 
    x_test, test_y = test_data 

    history = model.fit(
        x = {'input_ids': x_train['input_ids'],
            'attention_mask': x_train['attention_mask']},
        y = train_y,
        validation_data = [
            {'input_ids': x_test['input_ids'],
            'attention_mask': x_test['attention_mask']}, 
            test_y],
        epochs = int(epochs),
        batch_size = int(batch),
        callbacks = [callbacks])
    return model, history

def bert_load(transformer_name,transformer, model_path): 
    cust = {transformer_name: transformer}
    model = load_model(model_path, custom_objects = cust)
    return model

#LSTM
def lstm_build_sequences(data, max_words, max_len, tokenizer = None): 
    if tokenizer == None: #fit new tokenizer
        tokenizer = Tokenizer(num_words=max_words, lower=False)
        tokenizer.fit_on_texts(data)

    sequences = tokenizer.texts_to_sequences(data)
    sequences = pad_sequences(sequences, maxlen = max_len)
    return sequences, tokenizer

def lstm_build_embed_mat(emb_name, embeddings_index, word_index, max_words, embedding_dim = None):
    if embedding_dim == None:  
        if emb_name == 'fasttext':
            embedding_dim = 300
        else: 
            embedding_dim = len(embeddings_index[list(embeddings_index.keys())[0]]) #sampling 1st item of word vectors
    embedding_mat = np.zeros((max_words, embedding_dim))

    for word, i in word_index.items():
        if i < max_words: 
            embedding_vec = embeddings_index.get(word)
            if embedding_vec is not None: 
                embedding_mat[i] = embedding_vec
    return embedding_dim, embedding_mat

def lstm_processing(train_x, test_x, max_words, max_len): 
    train_x, tokenizer = lstm_build_sequences(train_x, max_words, max_len)
    test_x, tokenizer = lstm_build_sequences(test_x, max_words, max_len, tokenizer=tokenizer)
    word_index = tokenizer.word_index
    return train_x, test_x, word_index

def lstm_train(model, train_x, train_y, epochs, batch, callbacks):
    history = model.fit(train_x, np.array(train_y),
                        epochs=int(epochs),
                        batch_size=int(batch),
                        validation_split=0.1, 
                        callbacks=[callbacks])
    return model, history

#GENERAL
def get_scores(model, test_data, one_hot_encoded = 0, bert_input_format = 0): 
    test_x, test_y = test_data
    if bert_input_format: 
        testing_x = {'input_ids': test_x['input_ids'], 'attention_mask': test_x['attention_mask']}
    else: 
        testing_x = test_x
    #remove one-hot encoding
    if one_hot_encoded:
        test_y = list(map(np.argmax, test_y))
    
    y_preds = model.predict(testing_x)
    y_preds = list(map(np.argmax, y_preds))

    # target_names = [k for k,v in label_map.items()]
    accuracy = np.round(accuracy_score(test_y, y_preds), 4)
    f1 = np.round(f1_score(test_y, y_preds, average = 'macro'), 4)
    return y_preds, accuracy, f1

def get_reports(test_y, y_preds, label_map, test_one_hot = 1, pred_one_hot = 0):
    target_names = [k for k,v in label_map.items()]
    #remove one-hot encoding
    if test_one_hot:
        test_y = list(map(np.argmax, test_y))
    if pred_one_hot:
        y_preds = list(map(np.argmax, y_preds))

    clf_report = classification_report(test_y, y_preds, target_names = target_names, output_dict = True)
    conf_mat = confusion_matrix(test_y, y_preds)
    return clf_report, conf_mat

def tf_plot(history, model_name, emb_name, show = 1, save = 0): 
    acc = history.history['acc'][:-1]
    val_acc = history.history['val_acc'][:-1]
    loss = history.history['loss'][:-1]
    val_loss = history.history['val_loss'][:-1]
    epochs = range(0, len(acc))
    
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.plot(epochs, loss, 'rx', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    if save: 
        plt.savefig(f'{model_name}-{emb_name}-results.png')
    if show: 
        plt.show()

def save_clf_report(report, report_name):
    class_keys = list(report.keys())[:-3]
    class_metric_keys = ['precision', 'recall', 'f1-score', 'support']
    total_metric_keys = ['macro avg', 'weighted avg']

    report_data = pd.DataFrame(columns = ['class'] + class_metric_keys)
    for k, v in report.items():
        if isinstance(v, dict):
            row_dict = {}
            row_dict['class'] = k
            for k_, v_ in report[k].items():
                row_dict[k_] = v_
        else: #Accuracy metric
            row_dict['accuracy'] = k
            row_dict = {k: v}

        report_data = pd.concat([report_data, pd.DataFrame(row_dict, index=[0])])
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(f'results/{report_name}-results.csv', index = False)

def save_conf_mat(conf_mat, model_concat):
    np.savetxt(p.RESULTS_PATH+f'/{model_concat}-conf_mat.csv', conf_mat, delimiter=',')

def run_model(model, train_x, train_y, test_x, test_y, label_map): 
    #Model run - for SVM/XGB models
    model.fit(train_x, train_y)
    best_params = model.best_params_
    y_preds = model.predict(test_x)
    
    #Metrics
    target_names = [k for k,v in label_map.items()]
    accuracy = np.round(accuracy_score(test_y, y_preds), 4)
    f1 = np.round(f1_score(test_y, y_preds, average = 'macro'), 4)
    print("ACCURACY %: ", accuracy)
    print("F1 SCORE: ", f1)
    #Reports
    clf_report = classification_report(test_y, y_preds, target_names = target_names, output_dict = True)
    conf_mat = confusion_matrix(test_y, y_preds)
    return model, best_params, y_preds, accuracy, f1, clf_report, conf_mat

def store_results(model_name, emb_name, model, best_params, accuracy, f1_score, clf_report, conf_mat, tf_model = 0): 
    """Stores the following: 
        - model (sklearn or tensorflow) in /models dir
        - Precision/recall report + Confusion matrix in /results dir
        - overall accuracy + f1-score in /results/results.csv 
        - best model parameters in /results/best_params.csv
    """
    #SAVE MODEL
    model_concat = f'{model_name}-{emb_name}' if emb_name != '' else model_name
    model_path = f'models/{model_concat}'
    if tf_model: 
        model.save(f'{model_path}.h5', save_format='tf')
    else:
        with open(f'{model_path}.pkl', 'wb') as f: 
            pickle.dump(model, f)

    #SAVE ACCURACY + EVAL METRICS
    save_clf_report(clf_report, model_concat)
    save_conf_mat(conf_mat, model_concat)

    results_path = p.RESULTS_PATH+'results.csv'
    results_df = pd.read_csv(results_path)
    if model_concat in results_df['model_name'].values: 
        drop_mask = results_df['model_name'] == model_concat #Delete existing
        results_df.drop(results_df[drop_mask].index, inplace=True)
    model_results = pd.DataFrame({
                                'model_name': model_concat, 
                                'accuracy': accuracy, 
                                'f1': f1_score}, 
                                index = [0])
    results_df = pd.concat([results_df, model_results], ignore_index=True)
    results_df.to_csv(results_path, index=False)

    #SAVE BEST PARAMS
    params_path = p.RESULTS_PATH+'best_params.csv'
    params_df = pd.read_csv(params_path)
    if model_concat in params_df['model_name'].values: 
        drop_mask = params_df['model_name'] == model_concat #Delete existing
        params_df.drop(params_df[drop_mask].index, inplace=True)
    model_params = pd.DataFrame({
                                'model_name': model_concat, 
                                'params': str(best_params)}, 
                                index = [0])
    params_df = pd.concat([params_df, model_params], ignore_index=True)
    params_df.to_csv(params_path, index=False)


embedding_funcs = {
    'word2vec': build_word2vec_embeddings, 
    'glove': build_glove_embeddings, 
    'fasttext': build_fasttext_embeddings,
    'custom': build_custom_embeddings
}

def run_all(data, MODEL_NAME, OVERWRITE_FLAG): 
    pipeline, train_y, test_y, label_map = data
    if OVERWRITE_FLAG: 
        for emb_name, emb_func in embedding_funcs.items(): 
            print(f"RUNNING {MODEL_NAME.upper()} - {emb_name} embeddings")
            train_vecs, test_vecs = get_embedding_vecs(emb_name)
            model, best_params, y_preds, accuracy, f1, clf_report, conf_mat = run_model(pipeline, 
                                                                    train_vecs, 
                                                                    train_y, 
                                                                    test_vecs, 
                                                                    test_y, 
                                                                    label_map)
            store_results(MODEL_NAME, emb_name, model, best_params, accuracy, f1, clf_report, conf_mat)
    return model

#INTERPRETABILITY FUNCS
def explainer_get_weights(explainer, data_x, data_y, label_map, data_prop = 0.05, wt_threshold = 1.0, classification_type = None, n_samples = 10):
    class_names = list(label_map.keys())
    
    """
    classification_type = 
        'True' --> get word attributions for correctly classified instances only
        'False' --> get word attributions for INcorrectly classified instances only
        'None' --> get all word attributions
    """ 
    n = len(data_x); sample_rate = n_samples/n
    samples = [] #return a few samples for visualization
    results = {c:{'pos': [], 'neg': []} for c in class_names}
    # for c, c_idx in label_map.items(): 
    for x, y_true in zip(data_x, data_y): 
        if random.random() < data_prop: 
            #Get weights for instance; store sample
            w_weights = explainer(' '.join(x))
            if random.random() < (sample_rate):
                samples.append(w_weights)
                explainer.visualize()
            y_pred = explainer.predicted_class_index; print(y_pred)
            correctly_classified = y_pred == y_true
            #Break conditions
            if classification_type == True and not correctly_classified: 
                continue
            elif classification_type == False and correctly_classified: 
                continue
            #Store results
            neg_weights = [(w, wt) for (w, wt) in w_weights if wt < 0 and np.abs(wt) >= wt_threshold]
            pos_weights = [(w, wt) for (w, wt) in w_weights if wt > 0 and np.abs(wt) >= wt_threshold]
            c_name = class_names[y_pred]

            results[c_name]['neg'].append(neg_weights)
            results[c_name]['pos'].append(pos_weights)
    return results, samples

def explainer_get_common_words(results, label_map, top_n = 30): 
    class_names = list(label_map.keys())
    """For each class, determines the 'top_n' most heavily weighted positive and negative words"""
    results = {c:{'pos': [], 'neg': []} for c in class_names}
    for c in class_names:
        c_weights = {'pos': [], 'neg': []}
        for p in list(c_weights.keys()): 
            top_n = []
            polarity_weights = results[c][p]
            for x_weights in polarity_weights: 
                desc = True if p == 'pos' else False
                num_words = min(len(x_weights), top_n)
                top_n.extend(x_weights)
                top_n = sorted(top_n, key = lambda tup: tup[1], reverse = desc)[:num_words]
            c_weights[p] = top_n 
    return results