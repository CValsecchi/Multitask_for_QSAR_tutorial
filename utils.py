import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras import backend as K
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping
import time

MASK_VALUE = -1

class Settings:
    def __init__(self, data_train, data_test, layers=[100,100,10], act_fnc="relu", epochs=1000, batch_size=10, lr=0.001, opt="adam", drp=0.25):
        self.data_train = data_train
        self.data_test = data_test
        self.layers = layers
        self.act_fnc = act_fnc
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.opt = opt
        self.drp = drp

def masked_loss_function(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, MASK_VALUE), K.floatx())  #mask_value = -1
    return K.binary_crossentropy(y_true * mask, y_pred * mask)

def masked_accuracy(y_true, y_pred):
    total = K.sum(K.cast(K.not_equal(y_true, MASK_VALUE), dtype=K.floatx()))
    correct = K.sum(K.cast(K.equal(y_true, K.round(y_pred)), dtype=K.floatx()))
    return correct/(total+0.000001)
    
def prepare_data(data_train, data_test):
    data = data_train.fillna(MASK_VALUE)
    data_test = data_test.fillna(MASK_VALUE)

    start_fps = data.columns.get_loc('bit1')
    X = data.iloc[:,start_fps:].values
    y = data.iloc[:,3:start_fps].values
    X_test = data_test.iloc[:,start_fps:].values
    y_test = data_test.iloc[:,3:start_fps].values
    
    return X,y,X_test,y_test

def MTL_model_CV(Set):
    
    X,y,X_test,y_test = prepare_data(Set.data_train, Set.data_test)
    
    kf = KFold(n_splits=3, shuffle = True, random_state = 26)
    
    times = []
    y_val_true_all,y_val_pred_all = [],[]
    n_eps = []
    res = dict()
    
    for train_index, val_index in kf.split(X,y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        start_time = time.time()
        y_val_pred, n_ep = MTL_model(X_train, X_val, y_train, y_val, Set, 'yes')
        end_time = time.time()
        
        times.append(end_time-start_time)
        y_val_true_all.append(y_val)
        y_val_pred_all.append(y_val_pred)
        n_eps.append(n_ep)
    
    n_ep = int(np.mean(np.array(n_eps)))
    res['val_pred'] = np.concatenate(y_val_pred_all, axis=0)
    res['val_true'] = np.concatenate(y_val_true_all, axis=0)
    
    start_time = time.time()
    y_test_pred, n_ep = MTL_model(X, X_test, y, y_test, Set, 'no')
    end_time = time.time()
    
    res['test_pred'] = y_test_pred
    res['test_true'] = y_test
     
    return res
    
def do_opt(opt, lr):
    if opt == "adam":
        optim = keras.optimizers.Adam(learning_rate=lr)

    elif opt == "adamax":
        optim = keras.optimizers.Adamax(learning_rate=lr)

    elif opt == "sgd":
        optim = keras.optimizers.SGD(learning_rate=lr)

    elif opt == "rmsprop":
        optim = keras.optimizers.RMSprop(learning_rate=lr)
        
    return optim
    

def MTL_model(X_train, X_test, y_train, Set, early_stop="yes"):
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    optim = do_opt(Set.opt, Set.lr)
    n_y = y_train.shape[1]
    ############################
    inp = Input(shape=(X_train.shape[1],))
    hid = inp
    
    for l in Set.layers:
        hid = Dense(l, activation=Set.act_fnc, kernel_regularizer=keras.regularizers.L2(0.01))(hid)
        hid = Dropout(Set.drp)(hid)
    
    outs = []
    for i in range(n_y):
        outs.append(Dense(1, activation='sigmoid', name = "preds_"+str(i))(hid))

    model = keras.Model(inputs=inp, outputs=outs)
    model.compile(loss=masked_loss_function, optimizer=optim, metrics=[masked_accuracy])
    
    ############################
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    y_tr = [np.array(y_m).astype(np.float32) for y_m in y_train.T.tolist()]
    
    if early_stop == 'yes':
        history = model.fit(X_train_scaled, y_tr, epochs=Set.epochs, batch_size=int(Set.batch_size*n_x/100), callbacks=[es], verbose=0,
                 validation_data=(X_test, y_test))
    else:
        history = model.fit(X_train_scaled, y_tr, epochs=Set.epochs, batch_size=int(Set.batch_size*n_x/100), verbose=0,
             validation_data=(X_test, y_test))

    y_pred_test = model.predict(X_test_scaled)   
    y_pred_test = np.transpose(np.array(y_pred_test)[:,:,0])

    return y_pred_test, len(history.history['loss'])

def STL_models_CV(Set):

    X,y,X_test,y_test = prepare_data(Set.data_train, Set.data_test)
    
    res = dict()
    res['val_pred'],res['val_true'],res['test_pred'],res['test_true'] = [],[],[],[]
    
    ##########################
    for i in range(y.shape[1]):
        print(tasks[i])
        Xred = X[y[:,i]!=MASK_VALUE,:]
        yred = y[y[:,i]!=MASK_VALUE,i]
        Xtestred = X_test[y_test[:,i]!=MASK_VALUE,:]
        ytestred = y_test[y_test[:,i]!=MASK_VALUE,i]
        
        print(f" No samples = {len(Xred)} / {len(Xtestred)} actives = {np.sum(yred==1)/len(yred)} / {np.sum(ytestred==1)/len(ytestred)}")
        
        kf = KFold(n_splits=3, shuffle = True, random_state = 26)
    
        times = []
        y_val_true_all,y_val_pred_all = [],[]
        n_eps = []

        for train_index, val_index in kf.split(Xred,yred):
            Xtrain, X_val = Xred[train_index], Xred[val_index]
            ytrain, y_val = yred[train_index], yred[val_index]

            start_time = time.time()
            y_val_pred, n_ep = STL_model(Xtrain, X_val, ytrain, y_val, Set, 'yes')
            end_time = time.time()

            times.append(end_time-start_time)
            y_val_true_all.append(y_val)
            y_val_pred_all.append(y_val_pred)
            n_eps.append(n_ep)
        
        n_ep = int(np.mean(np.array(n_eps)))
        res[tasks[i]+' val_pred'] = np.concatenate(y_val_pred_all, axis=0)
        res[tasks[i]+' val_true'] = np.concatenate(y_val_true_all, axis=0)

        start_time = time.time()
        y_test_pred, n_ep = STL_model(Xred, Xtestred, yred, ytestred, Set, 'no')
        end_time = time.time()

        res[tasks[i]+' test_pred'] = y_test_pred

    return res

def STL_model(X_train, X_val, y_train, y_val, Set, early_stop="yes"):
    optim = do_opt(Set.opt, Set.lr)
        
    inp = Input(shape=(X_train.shape[1],))
    hid = inp
    for l in Set.layers:
        hid = Dense(l, activation=Set.act_fnc, kernel_regularizer=keras.regularizers.L2(0.01))(hid)
        hid = Dropout(Set.drp)(hid)
    out = Dense(1, activation='sigmoid', name = "preds")(hid)

    model = keras.Model(inputs=inp, outputs=out)
    model.compile(loss=K.binary_crossentropy, optimizer=optim, metrics=["accuracy"])
    #model.summary()
    
    n_x = X_train.shape[0]
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    
    if early_stop == 'yes':
        history = model.fit(X_train, y_train, epochs=Set.epochs, 
                            batch_size=int(Set.batch_size*n_x/100), 
                            callbacks=[es],verbose=0,
                            validation_data=(X_val, y_val))
    else:
        history = model.fit(X_train, y_train, epochs=Set.epochs, 
                        batch_size=int(Set.batch_size*n_x/100),verbose=0,
                        validation_data=(X_val, y_val))

    y_pred_val = model.predict(X_val)
    y_pred_val = y_pred_val.flatten()

    return y_pred_val, len(history.history['loss'])


def calc_output(res, thr_nrs, task):
    allsn, allsp, allner = [],[],[]
    for t in np.arange(res['val_pred'].shape[1]):
        y_pred_val_t = res['val_pred'][res['val_true'][:,t] != MASK_VALUE,t]
        y_val_t = res['val_true'][res['val_true'][:,t] != MASK_VALUE,t]
        preds_i = (y_pred_val_t>thr_nrs[t]*np.ones(y_pred_val_t.shape))*1
        sn, sp, ner = calc_class_param_mat(preds_i, np.array(y_val_t))
        allsn.append(sn)
        allsp.append(sp)
        allner.append(ner)
        
    val_res = pd.DataFrame([allsn, allsp, allner], columns = task, index = ['SN', 'SP', 'NER']).T
    
    allsn, allsp, allner = [],[],[]
    for t in np.arange(res['test_pred'].shape[1]):
        y_pred_val_t = res['test_pred'][res['test_true'][:,t] != MASK_VALUE,t]
        y_val_t = res['test_true'][res['test_true'][:,t] != MASK_VALUE,t]
        preds_i = (y_pred_val_t>thr_nrs[t]*np.ones(y_pred_val_t.shape))*1
        sn, sp, ner = calc_class_param_mat(preds_i, np.array(y_val_t))
        allsn.append(sn)
        allsp.append(sp)
        allner.append(ner)
        
    test_res = pd.DataFrame([allsn, allsp, allner], columns = task, index = ['SN', 'SP', 'NER']).T
    
    return val_res, test_res

def calc_output_stl(res, thr_nrs, task):
    allsn, allsp, allner = [],[],[]
    cnt = 0
    for t in task:
        y_pred_val_t = res[t+' val_pred']
        y_val_t = res[t+' val_true']
        preds_i = (y_pred_val_t>thr_nrs[cnt]*np.ones(y_pred_val_t.shape))*1
        sn, sp, ner = calc_class_param_mat(preds_i, np.array(y_val_t))
        allsn.append(sn)
        allsp.append(sp)
        allner.append(ner)
        cnt  += 1
        
    val_res = pd.DataFrame([allsn, allsp, allner], columns = task, index = ['SN', 'SP', 'NER']).T
    
    allsn, allsp, allner = [],[],[]
    cnt = 0
    for t in task:
        y_pred_val_t = res[t+' test_pred']
        y_val_t = res[t+' test_true']
        preds_i = (y_pred_val_t>thr_nrs[cnt]*np.ones(y_pred_val_t.shape))*1
        sn, sp, ner = calc_class_param_mat(preds_i, np.array(y_val_t))
        allsn.append(sn)
        allsp.append(sp)
        allner.append(ner)
        cnt += 1
        
    test_res = pd.DataFrame([allsn, allsp, allner], columns = task, index = ['SN', 'SP', 'NER']).T
    
    return val_res, test_res

def calc_optimal_thr(y_pred_val, y_val):
    thr_nrs=[]
    allsn = []
    allsp = []

    
    thr_nrs = []
    for t in np.arange(y_val.shape[1]):
        delta = []
        for i in np.arange(0,1,0.01):
            y_pred_val_t = y_pred_val[y_val[:,t] != MASK_VALUE,t]
            y_val_t = y_val[y_val[:,t] != MASK_VALUE,t]
            preds_i = (y_pred_val_t>i*np.ones(y_pred_val_t.shape))*1
            allsn, allsp, ner = calc_class_param_mat(preds_i, np.array(y_val_t))
            delta.append(np.abs(allsn-allsp))

        indexmin= np.argmin(np.array(delta))
        thr_nrs.append(np.arange(0,1,0.01)[indexmin])

    return np.array(thr_nrs)


def calc_class_param_mat(pred, true):

    tp = np.sum((pred==1)&(true==1))
    fn = np.sum((pred==0)&(true==1))
    fp = np.sum((pred==1)&(true==0))
    tn = np.sum((pred==0)&(true==0))
    
    sn = tp/(tp+fn+0.000000001)
    sp = tn/(tn+fp+0.000000001)
    ner = (sn+sp)/2

    return sn, sp, ner