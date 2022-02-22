#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# OTOTOXICITY clinical+SNPs prediction model
# =============================================================================
# Import standard libraries
import numpy as np
import pandas as pd
import time
import sys
from joblib import Parallel, delayed
from collections import Counter
import subprocess
import random
import heapq
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ParameterGrid,StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing

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

#return random SNPs
def return_random_lines_file(size, seed, file_path='/home/data/all_snps'):
    SIZE = size
    with open(file_path) as fin:
        random.seed(seed)
        sample = heapq.nlargest(SIZE, fin, key=lambda L: random.random())
    sample = list(map(lambda s: s.strip(), sample))
    return sample

#downsample (only for binary classification)
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

#returns snps from GWAS analysis
def inner_gwas(itrain_idx, fam_ID_dict, seed, sh_scriptpath="inner_gwas_NTX6.sh", outpath="/
home/data/intermediate"):
    ######## Inner GWAS##########
    snps_gwas=[]
    # File with train fids and iids (FID \t IID)
    inner_train_IIDs = itrain_idx
    intermidiate_train_ids = open(outpath+'/intermidiate_train_ids_'+str(seed), 'w')
    for k in {k:fam_ID_dict[str(k)] for k in inner_train_IIDs if str(k) in fam_ID_dict}:
        intermidiate_train_ids.write(fam_ID_dict[str(k)]+"\t"+str(k)+"\n")
    intermidiate_train_ids.close()
    # Run Inner GWAS
    sys.stdout.flush() #flush only job is done
    subprocess.call([sh_scriptpath, str(seed)])
    intermidiate_snp_ids = open(outpath+'/intermediate_snps_'+str(seed)+'.txt', 'r')
    for line in intermidiate_snp_ids:
        snps_gwas.append(line[:-1])
    intermidiate_snp_ids.close()
    return snps_gwas

#logistic regression w/ parameter optimization
def forward_fs_lr(Xitrain, Xval, yitrain, yval,best_features, all_snps,
                parameters={'C' : np.logspace(-4, 4, 20)}):
    dict_forward_fs={}
    for feat in all_snps:
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

def fs_ml(data_x, data_y, clinical_features,literature_snps,max_features, seed ,n_outer=5, n_inner=5, index_col_name='Unnamed: 0', data_x_path="/h
ome/data/X_data.csv", function=forward_fs_lr, imputation=True, normalize=False):
    inner_gwas_dict={} #save snps from each inner-gwas
    feat_selected=clinical_features.copy()
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
            inner_gwas_no=0
            # inner folds
            for itrain_idx, val_idx in inner_cv.split(X_train, y_train):
                y_itrain, y_val = y_train.iloc[itrain_idx], y_train.iloc[val_idx]
                #run inner-gwas only once for each outer and inner fold
                key_dict=str(seed)+'_'+str(i)+'_'+str(inner_gwas_no)
                if key_dict not in inner_gwas_dict:
                    gwas_snps_inner=inner_gwas(y_itrain.index,fam_ID_dict,seed)
                    inner_gwas_dict[key_dict]=gwas_snps_inner
                else:
                    gwas_snps_inner=inner_gwas_dict[key_dict]
                inner_gwas_no+=1
                random_snps=return_random_lines_file(25,seed)
                all_snps=gwas_snps_inner+literature_snps+random_snps
                all_snps = [x for x in all_snps if x not in feat_selected] #remove snps which were already selected
                if len(all_snps)==0:
                    return feat_selected
                #import only necessary columns from df
                fields=feat_selected+all_snps
                df = pd.read_csv(data_x_path, index_col=0, usecols=[index_col_name]+fields)
                df.index=df.index.astype(int)
                X_itrain_tmp, X_val_tmp = df.iloc[train_idx], df.iloc[train_idx]
                X_itrain_tmp, X_val_tmp = X_itrain_tmp.iloc[itrain_idx], X_itrain_tmp.iloc[val_idx]
                if imputation:
                    X_itrain_tmp, X_val_tmp = impute_data(X_itrain_tmp),impute_data(X_val_tmp)
                if normalize:
                    X_itrain_tmp, X_val_tmp =scale(X_itrain_tmp), scale(X_val_tmp)
                dict_forward_fs=function(X_itrain_tmp, X_val_tmp, y_itrain, y_val, feat_selected, all_snps)
                all_results_ML_tmp={k: v[0] for k, v in dict_forward_fs.items()}
                all_results_ML_models={k: v[1] for k, v in dict_forward_fs.items()}
                all_results_ML_train={}
                for key, value in all_results_ML_tmp.items():
                    all_results_ML_train[key]=value
                best_feat_temp=mode_dict(all_results_ML_train, 1) #choose feature with best roc-auc
                best_features_inner_lst.extend(best_feat_temp)
                best_clf, count = Counter(all_results_ML_models.values()).most_common(1)[0] #choose model selected most of the times
                best_models_inner_lst.append(best_clf)
                # del df
            best_feat_final=mode_list(best_features_inner_lst,1)
            print('best_feat_final:',best_feat_final)
            best_features_outer_lst.extend(best_feat_final) #append best features from each inner-fold
        #get feature to move forward (mode from each 5 outer fold)
        forward_feature=mode_list(best_features_outer_lst,1)
        print('forward_feature:',forward_feature)
        feat_selected.extend(forward_feature)
    return feat_selected

#forward selection_machine learning
def check_last_performaces(data_x, data_y, clinical_features,literature_snps,max_features, seed, n_outer=5, n_inner=5, index_col_name='Unnamed: 0'
, data_x_path="/home/neurotox/data/X_data.csv", function=forward_fs_rf, imputation=True, normalize=False, parameter
s={'n_estimators': [10, 50, 100, 150, 200]}, model=LogisticRegression(random_state=42, max_iter=10000, multi_class='auto')):
    print("seed: ",seed)
    all_results_ML={} #save roc-auc
    all_results_ML_inner={} #save roc-auc inner
    all_features={}  #save all features used in the model
    y_test_true={}
    y_test_pred={}
    y_test_index={}
    fields=fs_ml(data_x, data_y, clinical_features,literature_snps,max_features, seed ,n_outer, n_inner, index_col_name, data_x_path, function, im
putation, normalize)
    # evaluate performance on test fold
    df = pd.read_csv(data_x_path, index_col=0, usecols=[index_col_name]+fields)
    df.index=df.index.astype(int)
    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    for field_no in range(1,len(fields)+1):
        outer_roc_auc=[]
        inner_roc_auc=[]
        for i, (train_idx, test_idx) in enumerate(outer_cv.split(df[fields[0:field_no]], data_y)):
            X_train, X_test = df[fields[0:field_no]].iloc[train_idx], df[fields[0:field_no]].iloc[test_idx]
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
    all_features[str(seed)+"_"+str(len(X_test.columns))]=fields[0:field_no]
    return (all_results_ML,all_results_ML_inner, all_features, y_test_true, y_test_pred,y_test_index)

# =============================================================================
# get data
# =============================================================================
print("Load Data structures")

#import only index_col (only interested in sample ID)
X=pd.read_csv("/home/data/X_data.csv", index_col=0, usecols=[0])
X.index=X.index.astype(int)

#phenotype_ntx6
y=pd.read_csv("/home/data/y_data_NTX6.csv", index_col=0)
#PHENO 1
y['NTX6'] = y['NTX6'].map({0: 0, 1: 0,2: 1, 3: 1, 4:1})

#literature SNPs
with open('/home/results/VEP_literature/snps_high_drugresponse_literature.txt', 'r') as f:
    literature_snps = [line.strip() for line in f]
literature_snps=list(set(literature_snps))

# =============================================================================
# 1. Iterate over seeds
# =============================================================================
# ids dict with IIDs as keys and FIDs as values
QC_final_result = open('/home/inner_gwas/QC_data/A3302_final.fam', 'r')
fam_ID_dict={}
for line in QC_final_result:
    fam_ID_dict[line.split(" ")[1]]=line.split(" ")[0]

# =============================================================================
# MODEL : logistic regression
# =============================================================================
start = time.time()
np.warnings.filterwarnings('ignore')

label = "Oto NTX6"
print("Script that runs LR with the label: ",label)

#best clinical features
clinical_features=['Age_treatment1','treatment']

results=Parallel(n_jobs=32)(delayed(check_last_performaces)(X, y, clinical_features, literature_snps, len(clinical_features)+30, seed, function=fo
rward_fs_lr, normalize=True,parameters={'C' : np.logspace(-4, 4, 20)}, model=LogisticRegression(random_state=42, max_iter=10000, multi_class='auto
')) for seed in range(1,31))

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

end = time.time()
print('Complete in %.1f sec' %(end - start))

