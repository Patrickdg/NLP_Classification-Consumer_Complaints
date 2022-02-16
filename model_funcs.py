import processing as p

import os
import numpy as np
import pandas as pd
import pickle
import gensim
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
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

def get_embedding_vecs(emb_name): 
    paths = ['train_x_vecs.pkl', 'test_x_vecs.pkl']
    vecs = []
    for p in paths:
        vec_path = os.path.join('embedding_vecs', emb_name, p) 
        with open(vec_path, 'rb') as f: 
            vecs.append(pickle.load(f))

    return vecs

"""MODEL-SPECIFIC FUNCS======================================================"""
def bert_processing(train_x, test_x, train_y, test_y): 
    for data in [train_x, test_x]: 
        for i, l in enumerate(data): 
            data[i] = ' '.join(data[i])
    
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y) 

    return train_x, test_x, train_y, test_y

def bert_tokenize(model_name, tokenizer, train_x, test_x, max_length): 
    tokenizer = tokenizer.from_pretrained(model_name)
    tokenized = []
    for data in [train_x, test_x]: 
        tokenized_data = tokenizer(
            text=data,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding=True, 
            return_tensors='tf',
            return_token_type_ids = False,
            return_attention_mask = True,
            verbose = True)

        tokenized.append(tokenized_data)
    return tokenized

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

def generic_plot(epochs, acc, val_acc, loss, val_loss, model_name): 
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
    plt.savefig(f'{model_name}-results.png')
    plt.show()

def bert_plot(model_name, model, history, test_x, test_y, label_map): 
    #METRICS
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    generic_plot(epochs, acc, val_acc, loss, val_loss, model_name)
    #REPORTING
    testing_x = {'input_ids': test_x['input_ids'], 'attention_mask': test_x['attention_mask']}
    y_preds = model.predict(testing_x)
    num_classes = len(list(label_map.items()))
    for idx, l in enumerate(y_preds): 
        coded = np.zeros(num_classes)
        coded[np.argmax(l, axis = 0)] = 1.0
        y_preds[idx] = coded

    target_names = [k for k,v in label_map.items()]
    accuracy = np.round(accuracy_score(test_y, y_preds), 4)
    f1 = np.round(f1_score(test_y, y_preds, average = 'macro'), 4)

    return y_preds, accuracy, f1

def lstm_plot(model_name, model, history, test_x, test_y):
    acc = history.history['acc'][:-1]
    val_acc = history.history['val_acc'][:-1]
    loss = history.history['loss'][:-1]
    val_loss = history.history['val_loss'][:-1]
    epochs = range(0, len(acc))

    generic_plot(epochs, acc, val_acc, loss, val_loss, model_name)

    #TESTING instances
    y_pred_arrs = model.predict(test_x)
    y_preds = []
    for pred in y_pred_arrs: 
        y_preds.append(np.argmax(pred))
    accuracy = np.round(accuracy_score(test_y, y_preds), 4)
    f1 = np.round(f1_score(test_y, y_preds, average = 'macro'), 4)

    return y_preds, accuracy, f1

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
    #Save model
    model_concat = f'{model_name}-{emb_name}'
    model_path = f'models/{model_concat}'
    if tf_model: 
        if not os.path.exists(model_path): 
            os.mkdir(model_path)
        model.save(model_path)
    else:
        with open(f'{model_path}.pkl', 'wb') as f: 
            pickle.dump(model, f)

    #Record results
    save_clf_report(clf_report, model_concat)
    save_conf_mat(conf_mat, model_concat)
    results_path = p.RESULTS_PATH+'results.csv'
    results_df = pd.read_csv(results_path)
    if model_concat in results_df['model_name'].values: 
        #Delete existing instance
        drop_mask = results_df['model_name'] == model_concat
        results_df.drop(results_df[drop_mask].index, inplace=True)
    results_df = results_df.append({
        'model_name': model_concat, 
        'accuracy': accuracy, 
        'f1': f1_score}, 
        ignore_index=True)
    results_df.to_csv(results_path, index=False)

    #Best Params
    params_path = p.RESULTS_PATH+'best_params.csv'
    params_df = pd.read_csv(params_path)
    if model_concat in params_df['model_name'].values: 
        #Delete existing instance
        drop_mask = params_df['model_name'] == model_concat
        params_df.drop(params_df[drop_mask].index, inplace=True)
    params_df = params_df.append({
        'model_name': model_concat, 
        'params': str(best_params)},
        ignore_index = True)
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