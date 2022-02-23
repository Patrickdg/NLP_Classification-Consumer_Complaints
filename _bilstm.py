import model_funcs as mf
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Embedding
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD, Adam

"""ARCHITECTURE"""
def bilstm_model(embedding_dim, embedding_mat, hidden_layers = 0, hidden_units = 100): 
    model = Sequential()
    model.add(Embedding(MAX_WORDS, embedding_dim, input_length=MAX_LEN))
    # model.add(Bidirectional(LSTM(embedding_dim, activity_regularizer=l2(0.001),  kernel_regularizer=l2(0.001)))) #, kernel_regularizer=l2(0.01)
    # model.add(Bidirectional(LSTM(embedding_dim, return_sequences=True))) #, kernel_regularizer=l2(0.01)
    model.add(Bidirectional(LSTM(embedding_dim)))
    # for h in range(hidden_layers): 
    model.add(Dense(hidden_units, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.layers[0].set_weights([embedding_mat])
    model.layers[0].trainable = False

    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRIC)
    # model.summary()
    return model

if __name__ == "__main__": 
    train_x, test_x, train_y, test_y, label_map  = mf.get_train_test()

    #GENERAL PARAMS
    MODEL_NAME = 'bilstm'
    NUM_CLASSES = len(label_map)
    OVERWRITE_FLAGS = { #select which model-embedding combination to rerun+overwrite
        'word2vec': True, 
        'glove': False, 
        'fasttext': False,
        'custom': False
    }

    #TUNING PARAMS
    MAX_LEN = 300 # maximum length of input vectors
    MAX_WORDS = 10000 # top n words in vocab

    NUM_ADD_HIDDEN_LAYERS = 0
    NUM_HIDDEN_UNITS = 100
    EPOCHS = 2
    BATCH_SIZE = 32
    PATIENCE = 10
    MIN_DELTA = 0.0001

    LR = 1e-3
    MOMENTUM = 0.9
    DECAY = 1e-4
    LOSS = 'sparse_categorical_crossentropy'
    METRIC = ['acc']
    OPTIMIZER = Adam(learning_rate=LR, decay=DECAY)
    CALLBACKS = EarlyStopping(monitor='val_acc',
                            min_delta=MIN_DELTA,
                            patience=PATIENCE, 
                            restore_best_weights=True)
    PARAMS = {var: eval(var) for var in ['MAX_LEN',
                                        'MAX_WORDS',
                                        'NUM_ADD_HIDDEN_LAYERS',
                                        'BATCH_SIZE',
                                        'EPOCHS',
                                        'PATIENCE',
                                        'LR', 
                                        'MOMENTUM', 
                                        'DECAY'
                                        ]}
    #Data processing
    train_x, test_x, word_index = mf.lstm_processing(train_x, test_x, MAX_WORDS, MAX_LEN)
    num_classes = len(list(label_map.keys()))
    
    #Run model
    for emb_name, emb_func in mf.embedding_funcs.items(): 
        if OVERWRITE_FLAGS[emb_name]: 
            model_name = f'{MODEL_NAME}-{emb_name}'
            model_path = f'models/{model_name}'
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            print(f"RUNNING {MODEL_NAME.upper()} - {emb_name} embeddings")
            #Get embeddings
            emb_dict = mf.embedding_funcs
            emb_func = emb_dict[emb_name]; embeddings_index = emb_func()
            embedding_dim, embedding_mat = mf.lstm_build_embed_mat(emb_name, embeddings_index, word_index, MAX_WORDS)
            #Build, train model
            model = bilstm_model(embedding_dim, embedding_mat, MAX_WORDS, MAX_LEN)
            model, history = mf.lstm_train(model, train_x, train_y, EPOCHS, BATCH_SIZE, CALLBACKS)
            y_preds, accuracy, f1 = mf.get_scores(model, [test_x, test_y])
            clf_report, conf_mat = mf.get_reports(test_y, y_preds, label_map, test_one_hot=0, pred_one_hot=0)
            mf.store_results(MODEL_NAME, emb_name, model, PARAMS, accuracy, f1, clf_report, conf_mat, tf_model=1)
            mf.tf_plot(history, MODEL_NAME, emb_name, show = 0, save = 1)
