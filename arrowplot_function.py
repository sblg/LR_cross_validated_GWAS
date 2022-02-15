#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 13:41:54 2020

@author: sarga
"""
import matplotlib.pyplot as plt


#you need to give it:
#1) all_probas_: df with probability of belonging to class 1 (model with SNPs) - have id_sample in index
#2) all_probas_: df with probability of belonging to class 1 (model without SNPs) - have id_sample in index
#3) list with IDs of affected samples
#4) cutoff you want to use to consider sample to belong to one class (std = 0.5)

def arrow_plot(all_probas_, all_probas_without_SNPs, samples_affected, column_name=0,cutoff=0.5):
    all_y_axis = []
    for i in range(len(all_probas_)):
        all_y_axis.append(i)
    all_y_axis = list(range(len(all_y_axis)+80))

    norm = [(float(i)-min(all_y_axis))/(max(all_y_axis)-min(all_y_axis)) for i in all_y_axis]

    df_affected = all_probas_
    df_notAffected = all_probas_
    for i in all_probas_.index:
        if i in samples_affected:
            df_notAffected = df_notAffected.drop(i, axis=0)
        else:
            df_affected = df_affected.drop(i, axis=0)

    df_affected_without_SNPs = all_probas_without_SNPs
    df_notAffected_without_SNPs = all_probas_without_SNPs
    for i in all_probas_without_SNPs.index:
        if i in samples_affected:
            df_notAffected_without_SNPs = df_notAffected_without_SNPs.drop(i, axis=0)
        else:
            df_affected_without_SNPs = df_affected_without_SNPs.drop(i, axis=0)

    #start = 0
    #for i in range(len(all_probas_)):
    #    if i < len(df_affected):
    ##    print(all_probas_.values[i])
    #        plt.annotate('', xy = (df_affected.values[i], norm[i]),  xycoords = 'axes fraction', \
    #            xytext = (df_affected_without_SNPs.values[i], norm[i]), textcoords = 'axes fraction', fontsize = 8, \
    #            color = 'firebrick', arrowprops=dict(edgecolor='firebrick', arrowstyle = '->', shrinkA = 0, shrinkB = 0))
    #    else:
    #        plt.annotate('', xy = (df_notAffected.values[start], norm[i]),  xycoords = 'axes fraction', \
    #            xytext = (df_notAffected_without_SNPs.values[start], norm[i]), textcoords = 'axes fraction', fontsize = 8, \
    #            color = 'green', arrowprops=dict(edgecolor='green', arrowstyle = '->', shrinkA = 0, shrinkB = 0))
    #        start += 1

    df_notAffected_without_SNPs = df_notAffected_without_SNPs.sort_values(by=column_name)
    df_affected_without_SNPs = df_affected_without_SNPs.sort_values(by=column_name)

    new_index= list(df_notAffected_without_SNPs.index)
    df_notAffected = df_notAffected.reindex(new_index)

    new_index= list(df_affected_without_SNPs.index)
    df_affected = df_affected.reindex(new_index)


    ##FOR AFFECTED INDIVIDUALS
    SNPs_confused = []
    easily_predictable = []
    difficult_rescued = []
    difficult_not_rescued = []

    target = cutoff
    for i in range(len(df_affected)):
        if df_affected_without_SNPs.values[i] >= target and df_affected.values[i] < target:
            SNPs_confused.append(df_affected.index[i])
        if df_affected_without_SNPs.values[i] >= target and df_affected.values[i] >= target:
            easily_predictable.append(df_affected.index[i])
        if df_affected_without_SNPs.values[i] < target and df_affected.values[i] >= target:
            difficult_rescued.append(df_affected.index[i])
        if df_affected_without_SNPs.values[i] < target and df_affected.values[i] < target:
            difficult_not_rescued.append(df_affected.index[i])

    df_affected_SNPsconfused = df_affected
    df_affected_without_SNPs_SNPsconfused = df_affected_without_SNPs
    df_affected_easily_predictable = df_affected
    df_affected_without_SNPs_easily_predictable = df_affected_without_SNPs
    df_affected_ifficult_rescued = df_affected
    df_affected_without_SNPs_ifficult_rescued = df_affected_without_SNPs
    df_affected_difficult_not_rescued = df_affected
    df_affected_without_SNPs_difficult_not_rescued = df_affected_without_SNPs

    for i in df_affected.index:
        if i not in SNPs_confused:
            df_affected_SNPsconfused = df_affected_SNPsconfused.drop(i, axis=0)
            df_affected_without_SNPs_SNPsconfused = df_affected_without_SNPs_SNPsconfused.drop(i, axis=0)
        if i not in easily_predictable:
            df_affected_easily_predictable = df_affected_easily_predictable.drop(i, axis=0)
            df_affected_without_SNPs_easily_predictable = df_affected_without_SNPs_easily_predictable.drop(i, axis=0)
        if i not in difficult_rescued:
            df_affected_ifficult_rescued = df_affected_ifficult_rescued.drop(i, axis=0)
            df_affected_without_SNPs_ifficult_rescued = df_affected_without_SNPs_ifficult_rescued.drop(i, axis=0)
        if i not in difficult_not_rescued:
            df_affected_difficult_not_rescued = df_affected_difficult_not_rescued.drop(i, axis=0)
            df_affected_without_SNPs_difficult_not_rescued = df_affected_without_SNPs_difficult_not_rescued.drop(i, axis=0)
    # print("Affected SNPs confused:", list(df_affected_SNPsconfused.index))
    # print("Affected not rescued:", list(df_affected_difficult_not_rescued.index))
    print("Affected SNPs confused:", df_affected_SNPsconfused)
    print("Affected not rescued:", df_affected_difficult_not_rescued)

    ##FOR NON-AFFECTED INDIVIDUALS
    SNPs_confused_ = []
    easily_predictable_ = []
    difficult_rescued_ = []
    difficult_not_rescued_ = []

    for i in range(len(df_notAffected)):
        if df_notAffected_without_SNPs.values[i] < target and df_notAffected.values[i] >= target:
            SNPs_confused_.append(df_notAffected.index[i])
        if df_notAffected_without_SNPs.values[i] < target and df_notAffected.values[i] < target:
            easily_predictable_.append(df_notAffected.index[i])
        if df_notAffected_without_SNPs.values[i] >= target and df_notAffected.values[i] < target:
            difficult_rescued_.append(df_notAffected.index[i])
        if df_notAffected_without_SNPs.values[i] >= target and df_notAffected.values[i] >= target:
            difficult_not_rescued_.append(df_notAffected.index[i])


    df_notAffected_SNPsconfused_ = df_notAffected
    df_notAffected_without_SNPs_SNPsconfused_ = df_notAffected_without_SNPs
    df_notAffected_easily_predictable_ = df_notAffected
    df_notAffected_without_SNPs_easily_predictable_ = df_notAffected_without_SNPs
    df_notAffected_ifficult_rescued_ = df_notAffected
    df_notAffected_without_SNPs_ifficult_rescued_ = df_notAffected_without_SNPs
    df_notAffected_difficult_not_rescued_ = df_notAffected
    df_notAffected_without_SNPs_difficult_not_rescued_ = df_notAffected_without_SNPs


    for i in df_notAffected.index:
        if i not in SNPs_confused_:
            df_notAffected_SNPsconfused_ = df_notAffected_SNPsconfused_.drop(i, axis=0)
            df_notAffected_without_SNPs_SNPsconfused_ = df_notAffected_without_SNPs_SNPsconfused_.drop(i, axis=0)
        if i not in easily_predictable_:
            df_notAffected_easily_predictable_ = df_notAffected_easily_predictable_.drop(i, axis=0)
            df_notAffected_without_SNPs_easily_predictable_ = df_notAffected_without_SNPs_easily_predictable_.drop(i, axis=0)
        if i not in difficult_rescued_:
            df_notAffected_ifficult_rescued_ = df_notAffected_ifficult_rescued_.drop(i, axis=0)
            df_notAffected_without_SNPs_ifficult_rescued_ = df_notAffected_without_SNPs_ifficult_rescued_.drop(i, axis=0)
        if i not in difficult_not_rescued_:
            df_notAffected_difficult_not_rescued_ = df_notAffected_difficult_not_rescued_.drop(i, axis=0)
            df_notAffected_without_SNPs_difficult_not_rescued_ = df_notAffected_without_SNPs_difficult_not_rescued_.drop(i, axis=0)
    # print("Not affected SNPs confused:", list(df_notAffected_SNPsconfused_.index))
    # print("Not affected not rescued:", list(df_notAffected_difficult_not_rescued_.index))
    print("Not affected SNPs confused:", df_notAffected_SNPsconfused_)
    print("Not affected not rescued:", df_notAffected_difficult_not_rescued_)

    ##PLOT
    #change norm_num around to fit the legends in the right side
    fig = plt.figure(figsize = (10,10))
    plt.xlabel('Logistic regression Score')
    #plt.ylabel('PATIENTS')

    ax = fig.add_subplot(1,1,1)
    ax.set_ylabel(r'$\longleftarrow$PATIENTS$\longrightarrow$')

    ax.text(0.2,-0.1, "Not Affected",
            size = 20, ha = 'right', color='green')

    ax.text(0.8, -0.1, "Affected",
            size = 20, ha = 'left', color='red')

    ax.set_yticklabels([])

    #plt = plt.subplot(111)
    plt.axvline(target, linewidth=0.5, color='dimgrey',linestyle='dashed')
    #plt.axhline(0.6)
    #plt.axhline(0.8)

    start = 0
    start_1 = 0
    start_2 = 0
    start_3 = 0
    start_4 = 0
    start_5 = 0
    start_6 = 0
    norm_num = 5
    for i in range(len(all_probas_)+1):
        if i < len(df_notAffected_difficult_not_rescued_) and len(df_notAffected_difficult_not_rescued_)!=0 :
                # print(all_probas_.values[i])
            plt.annotate('', xy = (df_notAffected_difficult_not_rescued_.values[i], norm[norm_num]),  xycoords = 'axes fraction', \
                xytext = (df_notAffected_without_SNPs_difficult_not_rescued_.values[i], norm[norm_num]), textcoords = 'axes fraction', fontsize =
8, \
                color = 'green', arrowprops=dict(edgecolor='green', arrowstyle = '->', shrinkA = 0, shrinkB = 0))
            norm_num += 1
        if i == len(df_notAffected_difficult_not_rescued_) and len(df_notAffected_difficult_not_rescued_)!=0 :
            plt.axhline(norm[norm_num+5], linewidth=0.5, color='dimgrey',linestyle='dashed')
            norm_num += 10
            plt.text(1, (norm[norm_num+3]), s='Difficult, rescued:'+str(len(df_notAffected_ifficult_rescued_)), horizontalalignment='center', vert
icalalignment='center', transform=ax.transAxes)
            plt.text(1, (norm[norm_num-20]), s='Difficult, not rescued:'+str(len(df_notAffected_difficult_not_rescued_)), horizontalalignment='cen
ter', verticalalignment='center', transform=ax.transAxes)
        elif len(df_notAffected_difficult_not_rescued_) <= i <= len(df_notAffected_difficult_not_rescued_)+len(df_notAffected_ifficult_rescued_) a
nd len(df_notAffected_ifficult_rescued_) != 0:
            plt.annotate('', xy = (df_notAffected_ifficult_rescued_.values[start], norm[norm_num]),  xycoords = 'axes fraction', \
                xytext = (df_notAffected_without_SNPs_ifficult_rescued_.values[start], norm[norm_num]), textcoords = 'axes fraction', fontsize = 8
, \
                color = 'green', arrowprops=dict(edgecolor='green', arrowstyle = '->', shrinkA = 0, shrinkB = 0))
            start += 1
            norm_num += 1
        if i == len(df_notAffected_difficult_not_rescued_)+len(df_notAffected_ifficult_rescued_):
            plt.axhline(norm[norm_num+5], linewidth=0.5, color='dimgrey',linestyle='dashed')
            norm_num += 10
            plt.text(1, (norm[norm_num+5]), s='Easily predictable:'+str(len(df_notAffected_easily_predictable_)), horizontalalignment='center', ve
rticalalignment='center', transform=ax.transAxes)
        elif len(df_notAffected_difficult_not_rescued_)+len(df_notAffected_ifficult_rescued_) <= i <= len(df_notAffected_difficult_not_rescued_)+l
en(df_notAffected_ifficult_rescued_)+len(df_notAffected_easily_predictable_) and len(df_notAffected_easily_predictable_) != 0:
            plt.annotate('', xy = (df_notAffected_easily_predictable_.values[start_1], norm[norm_num]),  xycoords = 'axes fraction', \
                xytext = (df_notAffected_without_SNPs_easily_predictable_.values[start_1], norm[norm_num]), textcoords = 'axes fraction', fontsize
 = 8, \
                color = 'green', arrowprops=dict(edgecolor='green', arrowstyle = '->', shrinkA = 0, shrinkB = 0))
            start_1 += 1
            norm_num += 1
        if i == len(df_notAffected_difficult_not_rescued_)+len(df_notAffected_ifficult_rescued_)+len(df_notAffected_easily_predictable_):
            plt.axhline(norm[norm_num+5], linewidth=0.5, color='dimgrey',linestyle='dashed')
            norm_num += 10
            plt.text(1, (norm[norm_num+4]), s='Additional features confuse prediction:'+str(len(df_notAffected_SNPsconfused_)), horizontalalignmen
t='center', verticalalignment='center', transform=ax.transAxes)
        elif len(df_notAffected_difficult_not_rescued_)+len(df_notAffected_ifficult_rescued_)+len(df_notAffected_easily_predictable_) <= i <= len(
df_notAffected_difficult_not_rescued_)+len(df_notAffected_ifficult_rescued_)+len(df_notAffected_easily_predictable_)+len(df_notAffected_SNPsconfus
ed_) and len(df_notAffected_SNPsconfused_)!=0:
            plt.annotate('', xy = (df_notAffected_SNPsconfused_.values[start_2], norm[norm_num]),  xycoords = 'axes fraction', \
                xytext = (df_notAffected_without_SNPs_SNPsconfused_.values[start_2], norm[norm_num]), textcoords = 'axes fraction', fontsize = 8,
\
                color = 'green', arrowprops=dict(edgecolor='green', arrowstyle = '->', shrinkA = 0, shrinkB = 0))
            start_2 += 1
            norm_num += 1
        if i == len(df_notAffected_difficult_not_rescued_)+len(df_notAffected_ifficult_rescued_)+len(df_notAffected_easily_predictable_)+len(df_no
tAffected_SNPsconfused_):
            plt.axhline(norm[norm_num+5], linewidth=0.5, color='dimgrey',linestyle='dashed')
            norm_num += 10
            plt.text(1, (norm[norm_num+5]), s='Difficult, not rescued:'+str(len(df_affected_difficult_not_rescued)), horizontalalignment='center',
 verticalalignment='center', transform=ax.transAxes)
        elif len(df_notAffected) <= i <= len(df_notAffected)+len(df_affected_difficult_not_rescued):
            plt.annotate('', xy = (df_affected_difficult_not_rescued.values[start_3], norm[norm_num]),  xycoords = 'axes fraction', \
                xytext = (df_affected_without_SNPs_difficult_not_rescued.values[start_3], norm[norm_num]), textcoords = 'axes fraction', fontsize
= 8, \
                color = 'firebrick', arrowprops=dict(edgecolor='firebrick', arrowstyle = '->', shrinkA = 0, shrinkB = 0))
            start_3 += 1
            norm_num += 1
        if i == len(df_notAffected)+len(df_affected_difficult_not_rescued) and len(df_affected_difficult_not_rescued)!=0:
            plt.axhline(norm[norm_num+5], linewidth=0.5, color='dimgrey',linestyle='dashed')
            norm_num += 10
            plt.text(1, (norm[norm_num+5]), s='Difficult, rescued:'+str(len(df_affected_ifficult_rescued)), horizontalalignment='center', vertical
alignment='center', transform=ax.transAxes)
        elif len(df_notAffected)+len(df_affected_difficult_not_rescued) <= i <= len(df_notAffected)+len(df_affected_difficult_not_rescued)+len(df_
affected_ifficult_rescued) and len(df_affected_ifficult_rescued)!=0:
            plt.annotate('', xy = (df_affected_ifficult_rescued.values[start_4], norm[norm_num]),  xycoords = 'axes fraction', \
                xytext = (df_affected_without_SNPs_ifficult_rescued.values[start_4], norm[norm_num]), textcoords = 'axes fraction', fontsize = 8,
\
                color = 'firebrick', arrowprops=dict(edgecolor='firebrick', arrowstyle = '->', shrinkA = 0, shrinkB = 0))
            start_4 += 1
            norm_num += 1
        if i == len(df_notAffected)+len(df_affected_difficult_not_rescued)+len(df_affected_ifficult_rescued):
            plt.axhline(norm[norm_num+5], linewidth=0.5, color='dimgrey',linestyle='dashed')
            norm_num += 10
            plt.text(1, (norm[norm_num+3]), s='Easily predictable:'+str(len(df_affected_easily_predictable)), horizontalalignment='center', vertic
alalignment='center', transform=ax.transAxes)
        elif len(df_notAffected)+len(df_affected_difficult_not_rescued)+len(df_affected_ifficult_rescued) <= i <= len(df_notAffected)+len(df_affec
ted_difficult_not_rescued)+len(df_affected_ifficult_rescued)+len(df_affected_easily_predictable) and len(df_affected_easily_predictable)!=0:
            plt.annotate('', xy = (df_affected_easily_predictable.values[start_5], norm[norm_num]),  xycoords = 'axes fraction', \
                xytext = (df_affected_without_SNPs_easily_predictable.values[start_5], norm[norm_num]), textcoords = 'axes fraction', fontsize = 8
, \
                color = 'firebrick', arrowprops=dict(edgecolor='firebrick', arrowstyle = '->', shrinkA = 0, shrinkB = 0))
            start_5 += 1
            norm_num += 1
        elif i >= len(df_notAffected)+len(df_affected_difficult_not_rescued)+len(df_affected_ifficult_rescued)+len(df_affected_easily_predictable)
:
            plt.annotate('', xy = (df_affected_SNPsconfused.values[start_6], norm[norm_num]),  xycoords = 'axes fraction', \
                xytext = (df_affected_without_SNPs_SNPsconfused.values[start_6], norm[norm_num]), textcoords = 'axes fraction', fontsize = 8, \
                color = 'firebrick', arrowprops=dict(edgecolor='firebrick', arrowstyle = '->', shrinkA = 0, shrinkB = 0))
            start_6 += 1
            norm_num += 1
        if i == len(df_notAffected)+len(df_affected_difficult_not_rescued)+len(df_affected_ifficult_rescued)+len(df_affected_easily_predictable):
            plt.axhline(norm[norm_num+5], linewidth=0.5, color='dimgrey',linestyle='dashed')
            norm_num += 3
            plt.text(1, (norm[norm_num+5]), s='Additional features confuse prediction:'+str(len(df_affected_SNPsconfused)), horizontalalignment='c
enter', verticalalignment='center', transform=ax.transAxes)
