#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 09:32:00 2021

@author: max
"""

#set working directory
import os
os.chdir('/home/max/Documents/DRD/final_analyses_for_github_with_data')

#import libraries
from loading_RR import get_setup_ML, get_cat_vars, load_data
from BPt import * #v 1.3.4
import pandas as pd #v 1.1.5
import scipy.stats as stats#v 1.5.0
import statsmodels.api as sm #v 0.11.1
import numpy as np
import sklearn
from sklearn.impute import SimpleImputer

#%%
########################DRD Exclusions Differences Analysis##################
#build dataframes needed for analyses
data_all = load_data(get='all_noqc',val_check=3)
dat=data_all[0]
data_DRD_exclude=data_all[1]
chidf = pd.concat([dat, data_DRD_exclude], axis = 0, keys = ['inc', 'exc'])  
chidf['df'] = [x[0] for x in chidf.index]

#%%
#impute final sample data
for y in dat.columns[5:3153]:
    if(dat[y].dtype == np.float64 or dat[y].dtype == np.int64):
          imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
          imp_mean.fit(dat[y].to_numpy().reshape(-1, 1))
          dat[y] = imp_mean.transform(dat[y].to_numpy().reshape(-1, 1))
    else:
          imp_mean = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
          imp_mean.fit(dat[y].to_numpy().reshape(-1, 1))
          dat[y] = imp_mean.transform(dat[y].to_numpy().reshape(-1, 1))
          
#%%
#impute drd exclusion data
for y in data_DRD_exclude.columns[5:3153]:
    if(data_DRD_exclude[y].dtype == np.float64 or data_DRD_exclude[y].dtype == np.int64):
          imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
          imp_mean.fit(data_DRD_exclude[y].to_numpy().reshape(-1, 1))
          data_DRD_exclude[y] = imp_mean.transform(data_DRD_exclude[y].to_numpy().reshape(-1, 1))
    else:
          imp_mean = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
          imp_mean.fit(data_DRD_exclude[y].to_numpy().reshape(-1, 1))
          data_DRD_exclude[y] = imp_mean.transform(data_DRD_exclude[y].to_numpy().reshape(-1, 1))

#%%
#conduct ttests and chisquare tests
pvalues = list()

for y in dat.columns[5:3153]:
    if(dat[y].dtype == np.float64 or dat[y].dtype == np.int64):
        pval = sm.stats.ttest_ind(dat[y], data_DRD_exclude[y], 
                                   alternative='two-sided', usevar='pooled', 
                                   value=0)
        pvalues.append([y, pval[0], pval[1]])
    else:
        crosstab = pd.crosstab(chidf[y], chidf["df"])
        pval = stats.chi2_contingency(crosstab)
        
        pvalues.append([y, pval[0], pval[1]])
pvalues

#make ttest + chisquare results into df
results = pd.DataFrame(pvalues)
results.columns = ['variable', 't-stat/chi-square', 'pvalue']
#%%
results.to_csv("../t-test+chisquare.csv")


#%%
########################Puberty Analyses#########################
dat['puberty'].hist()
dat['puberty'].mean()
dat['puberty'].median()
dat['puberty'].value_counts()
dat['puberty'].value_counts(normalize=True)


#%%
data_DRD_exclude.shape
dat.shape
data_all[1]
dat.dropna(subset=["mean_indif"],axis=0,inplace=True)
dat.shape

from scipy.stats import pearsonr
pearsonr(dat['mean_indif'],dat['AUCmy'])


#%%
100/(1 + .05 *1)

#from paper
#V = A/(1 + k * D)
#D - delay
#A = amount
#V = value (i.e. indif point)

k = .05
V1_2 = 100/(1 + k * .25)
V1 = 100/(1 + k * 1)
V7 = 100/(1 + k * 7)
V30 = 100/(1 + k * 30)
V90 = 100/(1 + k * 90)
V365 = 100/(1 + k * 365)
V1825 = 100/(1 + k * 1825)

V1 = 100/(1 + k * D)
(V1/100) = 1 + k * D
((V1/100)-1) = k * D
((V1/100)-1)/D = k
k = ((V1/100)-1)/D

#%%
np.set_printoptions(suppress=True)
kdat = dat.loc[:,'ddis_scr_val_indif_point_6h':'ddis_scr_val_indif_pnt_5yr']
kmat = np.zeros([len(kdat), len(kdat.iloc[1])])
Ds = [.25, 1, 7, 30, 90, 365, 1825]
for sub in range(len(kdat)):
    i = 0
    for D in Ds:
        k = ((kdat.iloc[sub,i]/100)-1)/D
        kmat[sub,i] = k
        i += 1

kmat_log = np.log(-kmat)

#%%
D = (.25+ 1+ 7+ 30+ 90+ 365+ 1825)/7
mean_indiff = dat['mean_indif']
mkmat = np.zeros([len(mean_indiff), 1])
#for mean indifference point
for sub in range(len(mean_indiff)):
    k = ((mean_indiff.iloc[sub]/100)-1)/D
    mkmat[sub] = k
mkmat_log = np.log(-mkmat)



#%%
import matplotlib.pyplot as plt
plt.hist(mkmat_log)
plt.hist(mean_indiff)




