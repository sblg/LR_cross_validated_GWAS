#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 12:37:43 2020

@author: sarga
"""

# =============================================================================
# OTOTOXICITY clinical prediction model
# =============================================================================

# Import standard libraries
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import time
# Import Imputation libraries
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
# Import models libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ParameterGrid,StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
from collections import Counter

# =============================================================================
# functions
# =============================================================================
#returns scale df in range[0,1]
def scale(df):
    x = df.values #returns a numpy array
SGnewsetup_clinical.py...skipping...
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 12:37:43 2020

@author: sarga
"""

# =============================================================================
# OTOTOXICITY clinical prediction model
# =============================================================================

# Import standard libraries
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import time
# Import Imputation libraries
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
# Import models libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ParameterGrid,StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
from collections import Counter

# =============================================================================
# functions
# =============================================================================
#returns scale df in range[0,1]
def scale(df):
    x = df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
:...skipping...
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 12:37:43 2020

@author: sarga
"""

# =============================================================================
# OTOTOXICITY clinical prediction model
# =============================================================================

# Import standard libraries
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import time
# Import Imputation libraries
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
# Import models libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ParameterGrid,StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
from collections import Counter

# =============================================================================
# functions
# =============================================================================
#returns scale df in range[0,1]
def scale(df):
    x = df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_scaled = pd.DataFrame(x_scaled, columns=df.columns, index=df.index)
    return df_scaled

#returns imputed df (iterative approach)
def impute_data(df):
    imputer_iterative = IterativeImputer(missing_values=np.NaN, random_state=42,initial_strategy='most_frequent', max_iter=10000)
    X_data_impute = imputer_iterative.fit_transform(df)
    df_imputed=pd.DataFrame(X_data_impute, columns=df.columns, index=df.index)
    return df_imputed

#returns n modes in list
def mode_list(my_list, n):
    ct = dict(Counter(my_list))
    ct_sorted={k: v for k, v in sorted(ct.items(), reverse=True, key=lambda item: item[1])}
    return list(ct_sorted.keys())[0:n]

#returs n modes in dict
def mode_dict(my_dict, n):
    my_dict_sorted={k: v for k, v in sorted(my_dict.items(), reverse=True, key=lambda item: item[1])}
    return list(my_dict_sorted.keys())[0:n]

#downsample (only works for binary classification)
def downsample_major_class(X_train, y_train, major_class, seed):
    """
    X_train: Training data, input as a numpy array.
    y_train: Training labels, input as numpy array. Must have same N as X_train.
    major_class: integer, indicating the name of the major class in the y_train column.
    seed: Random seed, to make reproducable results.
    """
    # Indicies of each class' observations
    i_major = np.where(y_train == major_class)[0]
    i_major = y_train.index[i_major]
    minor_class = [i for i in [0, 1] if i != major_class][0]
    i_minor = np.where(y_train == minor_class)[0]
    i_minor = y_train.index[i_minor]
    # Number of observations in each class
    n_minor = len(i_minor)
    # For every observation of class 0, randomly sample from class 1 without replacement
    i_major_downsampled = np.random.RandomState(seed=seed).choice(i_major, size=n_minor, replace=False)
    # Join together class 0's target vector with the downsampled class 1's target vector
    all_y =         np.concatenate([i_minor,i_major_downsampled])
    y_train = y_train[y_train.index.isin(all_y)]
    # Concatenate the new training set
    X_train = X_train[X_train.index.isin(all_y)]
    return X_train, y_train

#logistic regression
def forward_fs_lr(Xitrain, Xval, yitrain, yval,best_features, all_features_try,
               parameters={'C' : np.logspace(-4, 4, 20)}):
    dict_forward_fs={}
    for feat in all_features_try:
        Xitrain_tmp=Xitrain[best_features+[feat]]
        Xitrain_tmp,yitrain=downsample_major_class(Xitrain_tmp, yitrain, 0, 42)
        Xval_tmp=Xval[best_features+[feat]]
        #returns a numpy array
        best_k, best_score = -1, -1
        clfs = {}
        # hyperparameter tuning
        param_grid = parameters
        for k in ParameterGrid(param_grid):
            pipe = LogisticRegression(random_state=42, max_iter=10000, multi_class='auto').set_params(**k)
            pipe.fit(Xitrain_tmp, yitrain.values.ravel())
            y_pred_inner = pipe.predict_proba(Xval_tmp)
            fpr, tpr, thresholds = roc_curve(yval, y_pred_inner[:, 1])
            score_mean = auc(fpr, tpr)
            if best_score < score_mean:
                best_k, best_score = str(k),score_mean
            clfs[str(k)] = pipe
        dict_forward_fs[feat]=best_score, clfs[best_k]
    return dict_forward_fs

#random forest w/ parameter optimization
def forward_fs_rf(Xitrain, Xval, yitrain, yval,best_features, all_features_try,
               parameters={'n_estimators': [10, 50, 100, 150, 200]}):
    dict_forward_fs={}
    for feat in all_features_try:
        Xitrain_tmp=Xitrain[best_features+[feat]]
        Xitrain_tmp,yitrain=downsample_major_class(Xitrain_tmp, yitrain, 0, 42)
        Xval_tmp=Xval[best_features+[feat]]
        #returns a numpy array
        best_k, best_score = -1, -1
        clfs = {}
        # hyperparameter tuning
        param_grid = parameters
        for k in ParameterGrid(param_grid):
            pipe = RandomForestClassifier(random_state=42).set_params(**k)
            pipe.fit(Xitrain_tmp, yitrain.values.ravel())
            y_pred_inner = pipe.predict_proba(Xval_tmp)
            fpr, tpr, thresholds = roc_curve(yval, y_pred_inner[:, 1])
            score_mean = auc(fpr, tpr)
            if best_score < score_mean:
                best_k, best_score = str(k),score_mean
            clfs[str(k)] = pipe
        dict_forward_fs[feat]=best_score, clfs[best_k]
    return dict_forward_fs

#artificial neural network w/ parameter optimization
def forward_fs_ann(Xitrain, Xval, yitrain, yval,best_features, all_features_try,
               parameters={
                    'hidden_layer_sizes': [(3,),(4,),(5,),(6,)],
                    'learning_rate_init':[0.001,0.01,0.05,0.1]}):
    dict_forward_fs={}
    for feat in all_features_try:
        Xitrain_tmp=Xitrain[best_features+[feat]]
        Xitrain_tmp,yitrain=downsample_major_class(Xitrain_tmp, yitrain, 0, 42)
        Xval_tmp=Xval[best_features+[feat]]
        #returns a numpy array
        best_k, best_score = -1, -1
        clfs = {}
        # hyperparameter tuning
        param_grid = parameters
        for k in ParameterGrid(param_grid):
            pipe = MLPClassifier(random_state=42,activation='relu',solver='adam', max_iter=100000).set_params(**k)
            pipe.fit(Xitrain_tmp, yitrain.values.ravel())
            y_pred_inner = pipe.predict_proba(Xval_tmp)
            fpr, tpr, thresholds = roc_curve(yval, y_pred_inner[:, 1])
            score_mean = auc(fpr, tpr)
            if best_score < score_mean:
                best_k, best_score = str(k),score_mean
            clfs[str(k)] = pipe
        dict_forward_fs[feat]=best_score, clfs[best_k]
    return dict_forward_fs

#forward selection_machine learning
def fs_ml(data_x, data_y, clinical_features, max_features, seed, n_outer=5, n_inner=5, function=forward_fs_rf, imputation=True, normalize=False):
    feat_selected=[]
    no_features=len(feat_selected)
    while no_features < max_features:
        no_features+=1
        print('Feature no.:', no_features)
        best_features_outer_lst=[]
        #outer fold
        outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
        inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
        for i, (train_idx, test_idx) in enumerate(outer_cv.split(data_x, data_y)):
            X_train = data_x.iloc[train_idx]
            y_train = data_y.iloc[train_idx]
            best_features_inner_lst=[]
            best_models_inner_lst=[]
            for itrain_idx, val_idx in inner_cv.split(X_train, y_train):
                # inner folds
                X_itrain, X_val = X_train.iloc[itrain_idx], X_train.iloc[val_idx]
                y_itrain, y_val = y_train.iloc[itrain_idx], y_train.iloc[val_idx]
                if imputation:
                    X_itrain_tmp, X_val_tmp = impute_data(X_itrain),impute_data(X_val)
                if normalize:
                    X_itrain_tmp, X_val_tmp =scale(X_itrain_tmp), scale(X_val_tmp)
                clinical_features = [x for x in clinical_features if x not in feat_selected] #remove featurres which were already selected
                dict_forward_fs=function(X_itrain_tmp, X_val_tmp, y_itrain, y_val, feat_selected, clinical_features)
                all_results_ML_tmp={k: v[0] for k, v in dict_forward_fs.items()}
                all_results_ML_models={k: v[1] for k, v in dict_forward_fs.items()}
                all_results_ML_train={}
                for key, value in all_results_ML_tmp.items():
                    all_results_ML_train[key]=value
                best_feat_temp=mode_dict(all_results_ML_train, 1) #choose feature with best roc-auc
                best_features_inner_lst.extend(best_feat_temp)
                best_clf, count = Counter(all_results_ML_models.values()).most_common(1)[0] #choose model selected most of the times
                best_models_inner_lst.append(best_clf)
            best_feat_final=mode_list(best_features_inner_lst,1)
            print('best_feat_final:',best_feat_final)
            best_features_outer_lst.extend(best_feat_final)
        #get feature to move forward (mode from each 5 outer fold)
        forward_feature=mode_list(best_features_outer_lst,1)
        print('forward_feature:',forward_feature)
        feat_selected.extend(forward_feature)
    return feat_selected

#forward selection_machine learning
def check_last_performaces(data_x, data_y, clinical_features,max_features, seed, n_outer=5, n_inner=5, function=forward_fs_rf, imputation=True, no
rmalize=False,parameters={'n_estimators': [10, 50, 100, 150, 200]},model=RandomForestClassifier(random_state=42)):
    print("seed: ",seed)
    all_results_ML={} #save roc-auc
    all_results_ML_inner={} #save roc-auc inner
    all_features={}  #save all features used in the model
    y_test_true={}
    y_test_pred={}
    y_test_index={}
    fields=fs_ml(data_x, data_y, clinical_features,max_features, seed ,n_outer, n_inner, function, imputation, normalize)
    # evaluate performance on test fold
    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    for field_no in range(1,len(fields)+1):
        outer_roc_auc=[]
        inner_roc_auc=[]
        for i, (train_idx, test_idx) in enumerate(outer_cv.split(data_x[fields[0:field_no]], data_y)):
            X_train, X_test = data_x[fields[0:field_no]].iloc[train_idx], data_x[fields[0:field_no]].iloc[test_idx]
            y_train, y_test = data_y.iloc[train_idx], data_y.iloc[test_idx]
            #parameter optimization
            best_k, best_score = -1, -1
            clfs = {}
            # hyperparameter tuning
            param_grid = parameters
            inner_scores=[]
            for k in ParameterGrid(param_grid):
                # inner folds
                for itrain_idx, val_idx in inner_cv.split(X_train, y_train):
                    X_itrain, X_val = X_train.iloc[itrain_idx], X_train.iloc[val_idx]
                    y_itrain, y_val = y_train.iloc[itrain_idx], y_train.iloc[val_idx]
                    if imputation:
                        X_itrain, X_val = impute_data(X_itrain),impute_data(X_val)
                    if normalize:
                        X_itrain, X_val =scale(X_itrain), scale(X_val)
                    pipe = model.set_params(**k)
                    pipe.fit(X_itrain, y_itrain.values.ravel())
                    y_pred_inner = pipe.predict_proba(X_val)
                    fpr, tpr, thresholds = roc_curve(y_val, y_pred_inner[:, 1])
                    inner_scores.append(auc(fpr, tpr))
                score_mean=np.mean(inner_scores)
                inner_roc_auc.append(score_mean)
                if best_score < score_mean:
                    best_k, best_score = str(k),score_mean
                clfs[str(k)] = pipe
            best_clf=clfs[best_k]
            if imputation:
                X_train, X_test = impute_data(X_train),impute_data(X_test)
            if normalize:
                X_train, X_test =scale(X_train), scale(X_test)
            X_train_tmp,y_train=downsample_major_class(X_train, y_train, 0, seed)
            best_clf.fit(X_train_tmp, y_train.values.ravel())
            #for testing
            y_pred = best_clf.predict_proba(X_test)
            #for testing
            fpr, tpr, thresholds = roc_curve(y_test, y_pred[:, 1])
            y_test_true[str(seed)+"_"+str(field_no)+"_"+str(i)]=list(y_test['NTX6'])
            y_test_pred[str(seed)+"_"+str(field_no)+"_"+str(i)]=y_pred[:, 1]
            y_test_index[str(seed)+"_"+str(field_no)+"_"+str(i)]=list(y_test.index)
            roc_auc = auc(fpr,tpr)
            outer_roc_auc.append(roc_auc)
        all_results_ML_inner["Features_roc_auc:"+str(len(X_test.columns))] = round(np.mean(inner_roc_auc), 4)
        all_results_ML["Features_roc_auc:"+str(len(X_test.columns))] = round(np.mean(outer_roc_auc), 4)
    all_features[str(seed)]=fields[0:field_no]
    return (all_results_ML,all_results_ML_inner, all_features, y_test_true, y_test_pred,y_test_index)

# =============================================================================
# get data
# =============================================================================
print("Load Data structures")

#clinical data
# X_T=pd.read_csv("/Users/sarga/Desktop/computerome_1/sarga/neurotox/data/X_data_T.csv", index_col=0, nrows=9)
X_T=pd.read_csv("/home/projects/pr_46457/people/sarga/neurotox/data/X_data_T.csv", index_col=0, nrows=9)
X=X_T.transpose()
X.index=X.index.astype(int)

#phenotype
y=pd.read_csv("/home/projects/pr_46457/people/sarga/neurotox/data/y_data_NTX6.csv", index_col=0)
#PHENO 1
y['NTX6'] = y['NTX6'].map({0: 0, 1: 0,2: 1, 3: 1, 4:1})
# #PHENO 2
# y['NTX6'] = y['NTX6'].map({0: 0, 1: 0,2: 0, 3: 1, 4:1})

# =============================================================================
# MODEL : logistic regression
# =============================================================================
start = time.time()
np.warnings.filterwarnings('ignore')

label = "Oto NTX6"
print("Script that runs LR with the label: ",label)

clinical_features=list(X.columns)
results=Parallel(n_jobs=32)(delayed(check_last_performaces)(X, y, clinical_features,len(clinical_features), seed, function=forward_fs_lr, normaliz
e=True,parameters={'C' : np.logspace(-4, 4, 20)}, model=LogisticRegression(random_state=42, max_iter=10000, multi_class='auto')) for seed in range
(1,31))

results_roc_auc=list()
results_roc_auc_inner=list()
results_feat={}
y_test_true={}
y_test_pred={}
y_test_index={}
for i_seed in results:
    results_roc_auc.append(i_seed[0])
    results_roc_auc_inner.append(i_seed[1])
    results_feat.update(i_seed[2])
    y_test_true.update(i_seed[3])
    y_test_pred.update(i_seed[4])
    y_test_index.update(i_seed[5])

results_roc_auc_pd=pd.DataFrame(results_roc_auc)
results_roc_auc_inner_pd=pd.DataFrame(results_roc_auc_inner)

# results_roc_auc_pd.to_csv("/home/projects/pr_46457/people/sarga/neurotox/results/forward_select_training/setup4_final/NTX6/pheno1/fs_lr_30seeds_
clinical.txt")
# results_roc_auc_inner_pd.to_csv("/home/projects/pr_46457/people/sarga/neurotox/results/forward_select_training/setup4_final/NTX6/pheno1/fs_lr_30
seeds_clinical_training.txt")
# pd.DataFrame.from_dict(data=results_feat, orient='index').to_csv('/home/projects/pr_46457/people/sarga/neurotox/results/forward_select_training/
setup4_final/NTX6/pheno1/fs_lr_30seeds_clinical_features.txt')
# pd.DataFrame.from_dict(data=y_test_true, orient='index').to_csv('/home/projects/pr_46457/people/sarga/neurotox/results/forward_select_training/s
etup4_final/NTX6/pheno1/fs_lr_30seeds_clinical_ytest.txt')
# pd.DataFrame.from_dict(data=y_test_pred, orient='index').to_csv('/home/projects/pr_46457/people/sarga/neurotox/results/forward_select_training/s
etup4_final/NTX6/pheno1/fs_lr_30seeds_clinical_ypred.txt')
# pd.DataFrame.from_dict(data=y_test_index, orient='index').to_csv('/home/projects/pr_46457/people/sarga/neurotox/results/forward_select_training/
setup4_final/NTX6/pheno1/fs_lr_30seeds_clinical_y_index.txt')

end = time.time()
print('No.features total', len(X.columns))
print('Complete in %.1f sec' %(end - start))

# # =============================================================================
# # MODEL : RANDOM FOREST
# # =============================================================================
# start = time.time()
# np.warnings.filterwarnings('ignore')

# label = "Oto  "
# print("Script that runs RF with the label: ",label)

# clinical_features=list(X.columns)
# results=Parallel(n_jobs=32)(delayed(check_last_performaces)(X, y, clinical_features,len(clinical_features), seed) for seed in range(1,31))

# results_roc_auc=list()
# results_roc_auc_inner=list()
# results_feat={}
# y_test_true={}
# y_test_pred={}
# y_test_index={}
# for i_seed in results:
#     results_roc_auc.append(i_seed[0])
#     results_roc_auc_inner.append(i_seed[1])
#     results_feat.update(i_seed[2])
#     y_test_true.update(i_seed[3])
#     y_test_pred.update(i_seed[4])
#     y_test_index.update(i_seed[5])

# results_roc_auc_pd=pd.DataFrame(results_roc_auc)
# results_roc_auc_inner_pd=pd.DataFrame(results_roc_auc_inner)

# results_roc_auc_pd.to_csv("/home/projects/pr_46457/people/sarga/neurotox/results/forward_select_training/setup4_final/NTX6/pheno1/fs_rf_30seeds_
clinical.txt")
# results_roc_auc_inner_pd.to_csv("/home/projects/pr_46457/people/sarga/neurotox/results/forward_select_training/setup4_final/NTX6/pheno1/fs_rf_30
seeds_clinical_training.txt")
# pd.DataFrame.from_dict(data=results_feat, orient='index').to_csv('/home/projects/pr_46457/people/sarga/neurotox/results/forward_select_training/
setup4_final/NTX6/pheno1/fs_rf_30seeds_clinical_features.txt')
# pd.DataFrame.from_dict(data=y_test_true, orient='index').to_csv('/home/projects/pr_46457/people/sarga/neurotox/results/forward_select_training/s
etup4_final/NTX6/pheno1/fs_rf_30seeds_clinical_ytest.txt')
# pd.DataFrame.from_dict(data=y_test_pred, orient='index').to_csv('/home/projects/pr_46457/people/sarga/neurotox/results/forward_select_training/s
etup4_final/NTX6/pheno1/fs_rf_30seeds_clinical_ypred.txt')
# pd.DataFrame.from_dict(data=y_test_index, orient='index').to_csv('/home/projects/pr_46457/people/sarga/neurotox/results/forward_select_training/
setup4_final/NTX6/pheno1/fs_rf_30seeds_clinical_y_index.txt')

# end = time.time()
# print('No.features total', len(X.columns))
# print('Complete in %.1f sec' %(end - start))

# # =============================================================================
# # MODEL : ANN
# # =============================================================================
# start = time.time()
# np.warnings.filterwarnings('ignore')

# label = "Oto NTX6"
# print("Script that runs ANN with the label: ",label)

# clinical_features=list(X.columns)
# results=Parallel(n_jobs=32)(delayed(check_last_performaces)(X, y, clinical_features,len(clinical_features), seed, function=forward_fs_ann, norma
lize=True, parameters={'hidden_layer_sizes': [(3,),(4,),(5,),(6,)],'learning_rate_init':[0.001,0.01,0.05,0.1]}, model=MLPClassifier(random_state=4
2,activation='relu',solver='adam', max_iter=100000)) for seed in range(1,31))

# results_roc_auc=list()
# results_roc_auc_inner=list()
# results_feat={}
# y_test_true={}
# y_test_pred={}
# y_test_index={}
# for i_seed in results:
#     results_roc_auc.append(i_seed[0])
#     results_roc_auc_inner.append(i_seed[1])
#     results_feat.update(i_seed[2])
#     y_test_true.update(i_seed[3])
#     y_test_pred.update(i_seed[4])
#     y_test_index.update(i_seed[5])

# results_roc_auc_pd=pd.DataFrame(results_roc_auc)
# results_roc_auc_inner_pd=pd.DataFrame(results_roc_auc_inner)

# results_roc_auc_pd.to_csv("/home/projects/pr_46457/people/sarga/neurotox/results/forward_select_training/setup4_final/NTX6/pheno1/fs_ann_30seeds
_clinical.txt")
# results_roc_auc_inner_pd.to_csv("/home/projects/pr_46457/people/sarga/neurotox/results/forward_select_training/setup4_final/NTX6/pheno1/fs_ann_3
0seeds_clinical_training.txt")
# pd.DataFrame.from_dict(data=results_feat, orient='index').to_csv('/home/projects/pr_46457/people/sarga/neurotox/results/forward_select_training/
setup4_final/NTX6/pheno1/fs_ann_30seeds_clinical_features.txt')
# pd.DataFrame.from_dict(data=y_test_true, orient='index').to_csv('/home/projects/pr_46457/people/sarga/neurotox/results/forward_select_training/s
etup4_final/NTX6/pheno1/fs_ann_30seeds_clinical_ytest.txt')
# pd.DataFrame.from_dict(data=y_test_pred, orient='index').to_csv('/home/projects/pr_46457/people/sarga/neurotox/results/forward_select_training/s
etup4_final/NTX6/pheno1/fs_ann_30seeds_clinical_ypred.txt')
# pd.DataFrame.from_dict(data=y_test_index, orient='index').to_csv('/home/projects/pr_46457/people/sarga/neurotox/results/forward_select_training/
setup4_final/NTX6/pheno1/fs_ann_30seeds_clinical_y_index.txt')

# end = time.time()
# print('No.features total', len(X.columns))
# print('Complete in %.1f sec' %(end - start))

# # =============================================================================
# # RANDOM models
# # =============================================================================
# def shuffle(df, n=1, axis=0):
#     df = df.copy()
#     for _ in range(n):
#         np.random.seed(10)
#         df.apply(np.random.shuffle, axis=axis)
#     return df

# # =============================================================================
# # VERSION 1
# # =============================================================================
# y.index=y.index.astype(str)
# y = shuffle(y)
# y.index=y.index.astype(int)

# # =============================================================================
# # MODEL : logistic regression
# # =============================================================================
# start = time.time()
# np.warnings.filterwarnings('ignore')

# label = "Oto NTX6"
# print("Script that runs LR with the label: ",label)

# clinical_features=list(X.columns)
# results=Parallel(n_jobs=32)(delayed(check_last_performaces)(X, y, clinical_features,len(clinical_features), seed, function=forward_fs_lr, normal
ize=True,parameters={'C' : np.logspace(-4, 4, 20)}, model=LogisticRegression(random_state=42, max_iter=10000, multi_class='auto')) for seed in ran
ge(1,31))

# results_roc_auc=list()
# for i_seed in results:
#     results_roc_auc.append(i_seed[0])

# results_roc_auc_pd=pd.DataFrame(results_roc_auc)

# results_roc_auc_pd.to_csv("/home/projects/pr_46457/people/sarga/neurotox/results/forward_select_training/setup4_final/NTX6/pheno1/fs_lr_30seeds_
clinical_randomY.txt")

# end = time.time()
# print('No.features total', len(X.columns))
# print('Complete in %.1f sec' %(end - start))

# # =============================================================================
# # MODEL : RANDOM FOREST
# # =============================================================================
# start = time.time()
# np.warnings.filterwarnings('ignore')

# label = "Oto  "
# print("Script that runs RF with the label: ",label)

# clinical_features=list(X.columns)
# results=Parallel(n_jobs=32)(delayed(check_last_performaces)(X, y, clinical_features,len(clinical_features), seed) for seed in range(1,31))

# results_roc_auc=list()
# for i_seed in results:
#     results_roc_auc.append(i_seed[0])

# results_roc_auc_pd=pd.DataFrame(results_roc_auc)

# results_roc_auc_pd.to_csv("/home/projects/pr_46457/people/sarga/neurotox/results/forward_select_training/setup4_final/NTX6/pheno1/fs_rf_30seeds_
clinical_randomY.txt")

# end = time.time()
# print('No.features total', len(X.columns))
# print('Complete in %.1f sec' %(end - start))

# # =============================================================================
# # MODEL : ANN
# # =============================================================================
# start = time.time()
# np.warnings.filterwarnings('ignore')

# label = "Oto NTX6"
# print("Script that runs ANN with the label: ",label)

# clinical_features=list(X.columns)
# results=Parallel(n_jobs=32)(delayed(check_last_performaces)(X, y, clinical_features,len(clinical_features), seed, function=forward_fs_ann, norma
lize=True, parameters={'hidden_layer_sizes': [(3,),(4,),(5,),(6,)],'learning_rate_init':[0.001,0.01,0.05,0.1]}, model=MLPClassifier(random_state=4
2,activation='relu',solver='adam', max_iter=100000)) for seed in range(1,31))

# results_roc_auc=list()
# for i_seed in results:
#     results_roc_auc.append(i_seed[0])

# results_roc_auc_pd=pd.DataFrame(results_roc_auc)

# results_roc_auc_pd.to_csv("/home/projects/pr_46457/people/sarga/neurotox/results/forward_select_training/setup4_final/NTX6/pheno1/fs_ann_30seeds
_clinical_randomY.txt")

# end = time.time()
# print('No.features total', len(X.columns))
# print('Complete in %.1f sec' %(end - start))

# # =============================================================================
# #  VERSION 2
# # =============================================================================
# X.index=X.index.astype(str)
# X = shuffle(X)
# X.index=X.index.astype(int)

# # =============================================================================
# # MODEL : logistic regression
# # =============================================================================
# start = time.time()
# np.warnings.filterwarnings('ignore')

# label = "Oto NTX6"
# print("Script that runs LR with the label: ",label)

# clinical_features=list(X.columns)
# results=Parallel(n_jobs=32)(delayed(check_last_performaces)(X, y, clinical_features,len(clinical_features), seed, function=forward_fs_lr, normal
ize=True,parameters={'C' : np.logspace(-4, 4, 20)}, model=LogisticRegression(random_state=42, max_iter=10000, multi_class='auto')) for seed in ran
ge(1,31))

# results_roc_auc=list()
# for i_seed in results:
#     results_roc_auc.append(i_seed[0])

# results_roc_auc_pd=pd.DataFrame(results_roc_auc)

# results_roc_auc_pd.to_csv("/home/projects/pr_46457/people/sarga/neurotox/results/forward_select_training/setup4_final/NTX6/pheno1/fs_lr_30seeds_
clinical_randomX.txt")

# end = time.time()
# print('No.features total', len(X.columns))
# print('Complete in %.1f sec' %(end - start))

# # =============================================================================
# # MODEL : RANDOM FOREST
# # =============================================================================
# start = time.time()
# np.warnings.filterwarnings('ignore')

# label = "Oto  "
# print("Script that runs RF with the label: ",label)

# clinical_features=list(X.columns)
# results=Parallel(n_jobs=32)(delayed(check_last_performaces)(X, y, clinical_features,len(clinical_features), seed) for seed in range(1,31))

# results_roc_auc=list()
# for i_seed in results:
#     results_roc_auc.append(i_seed[0])

# results_roc_auc_pd=pd.DataFrame(results_roc_auc)

# results_roc_auc_pd.to_csv("/home/projects/pr_46457/people/sarga/neurotox/results/forward_select_training/setup4_final/NTX6/pheno1/fs_rf_30seeds_
clinical_randomX.txt")

# end = time.time()
# print('No.features total', len(X.columns))
# print('Complete in %.1f sec' %(end - start))

# # =============================================================================
# # MODEL : ANN
# # =============================================================================
# start = time.time()
# np.warnings.filterwarnings('ignore')

# label = "Oto NTX6"
# print("Script that runs ANN with the label: ",label)

# clinical_features=list(X.columns)
# results=Parallel(n_jobs=32)(delayed(check_last_performaces)(X, y, clinical_features,len(clinical_features), seed, function=forward_fs_ann, norma
lize=True, parameters={'hidden_layer_sizes': [(3,),(4,),(5,),(6,)],'learning_rate_init':[0.001,0.01,0.05,0.1]}, model=MLPClassifier(random_state=4
2,activation='relu',solver='adam', max_iter=100000)) for seed in range(1,31))

# results_roc_auc=list()
# for i_seed in results:
#     results_roc_auc.append(i_seed[0])

# results_roc_auc_pd=pd.DataFrame(results_roc_auc)

# results_roc_auc_pd.to_csv("/home/projects/pr_46457/people/sarga/neurotox/results/forward_select_training/setup4_final/NTX6/pheno1/fs_ann_30seeds
_clinical_randomX.txt")

# end = time.time()
# print('No.features total', len(X.columns))
# print('Complete in %.1f sec' %(end - start))


# # =============================================================================
# # VERSION 3
# # =============================================================================
# np.random.seed(42)
# X = pd.DataFrame(np.random.randint(0,100,size=(393, 9)))
# y = pd.DataFrame(np.random.randint(0,2,size=(393, 1)))
# y.columns = ['NTX6']
# # y[0].value_counts()


# # =============================================================================
# # MODEL : logistic regression
# # =============================================================================
# start = time.time()
# np.warnings.filterwarnings('ignore')

# label = "Oto NTX6"
# print("Script that runs LR with the label: ",label)

# clinical_features=list(X.columns)
# results=Parallel(n_jobs=32)(delayed(check_last_performaces)(X, y, clinical_features,len(clinical_features), seed, function=forward_fs_lr, normal
ize=True,parameters={'C' : np.logspace(-4, 4, 20)}, model=LogisticRegression(random_state=42, max_iter=10000, multi_class='auto')) for seed in ran
ge(1,31))

# results_roc_auc=list()
# for i_seed in results:
#     results_roc_auc.append(i_seed[0])

# results_roc_auc_pd=pd.DataFrame(results_roc_auc)

# results_roc_auc_pd.to_csv("/home/projects/pr_46457/people/sarga/neurotox/results/forward_select_training/setup4_final/NTX6/pheno1/fs_lr_30seeds_
clinical_randomDF.txt")

# end = time.time()
# print('No.features total', len(X.columns))
# print('Complete in %.1f sec' %(end - start))

# # =============================================================================
# # MODEL : RANDOM FOREST
# # =============================================================================
# start = time.time()
# np.warnings.filterwarnings('ignore')

# label = "Oto  "
# print("Script that runs RF with the label: ",label)

# clinical_features=list(X.columns)
# results=Parallel(n_jobs=32)(delayed(check_last_performaces)(X, y, clinical_features,len(clinical_features), seed) for seed in range(1,31))

# results_roc_auc=list()
# for i_seed in results:
#     results_roc_auc.append(i_seed[0])

# results_roc_auc_pd=pd.DataFrame(results_roc_auc)

# results_roc_auc_pd.to_csv("/home/projects/pr_46457/people/sarga/neurotox/results/forward_select_training/setup4_final/NTX6/pheno1/fs_rf_30seeds_
clinical_randomDF.txt")

# end = time.time()
# print('No.features total', len(X.columns))
# print('Complete in %.1f sec' %(end - start))

# # =============================================================================
# # MODEL : ANN
# # =============================================================================
# start = time.time()
# np.warnings.filterwarnings('ignore')

# label = "Oto NTX6"
# print("Script that runs ANN with the label: ",label)

# clinical_features=list(X.columns)
# results=Parallel(n_jobs=32)(delayed(check_last_performaces)(X, y, clinical_features,len(clinical_features), seed, function=forward_fs_ann, norma
lize=True, parameters={'hidden_layer_sizes': [(3,),(4,),(5,),(6,)],'learning_rate_init':[0.001,0.01,0.05,0.1]}, model=MLPClassifier(random_state=4
2,activation='relu',solver='adam', max_iter=100000)) for seed in range(1,31))

# results_roc_auc=list()
# for i_seed in results:
#     results_roc_auc.append(i_seed[0])

# results_roc_auc_pd=pd.DataFrame(results_roc_auc)

# results_roc_auc_pd.to_csv("/home/projects/pr_46457/people/sarga/neurotox/results/forward_select_training/setup4_final/NTX6/pheno1/fs_ann_30seeds
_clinical_randomDF.txt")

# end = time.time()
# print('No.features total', len(X.columns))
# print('Complete in %.1f sec' %(end - start))
