#set working directory
import os
os.chdir('/home/max/Dropbox/ABCD/DRD/DRD')

import pandas as pd #v 1.1.5
import numpy as np #v 1.18.1
import matplotlib.pyplot as plt #v 3.2.2
import matplotlib.ticker as mtick #v 3.2.2
from numpy import matrix #v 1.18.1

###############set up##############
#define functions

def zero_floor_only(x):
    y=[0 if i < 0 else i for i in list(x)]
    return y
def zero_floor(x):
    y=[0 if i < 0 else i for i in list(x)]
    yy = [i * 100 for i in y]
    return yy
def zero_floor_diff(x):
    yy = [i * 100 for i in x]
    return yy

#make tables is used when you have mri only, covariate only, and mri + covariate
def make_tables(datas, output_name):
    result_table = pd.DataFrame()
    covs_table = pd.DataFrame()
    all_table = pd.DataFrame()
    diff_table = pd.DataFrame()

    EN = datas[0].transpose()
    RF = datas[1].transpose()
    LGB = datas[2].transpose()
    SVM = datas[3].transpose()
    result_table['Elastic Net'] = EN[0].reset_index(drop=True)
    result_table['Random Forest'] = RF[0].reset_index(drop=True)
    result_table['Gradient Boosting'] = LGB[0].reset_index(drop=True)
    result_table['Support Vector Machine'] = SVM[0].reset_index(drop=True)
    result_table.index = ['full_sample','smri','dmri','rs','nback','sst','mid']
    result_table.to_csv(output_name)

#make tables 2 is for use when you only have mri variable analyses (i.e., RFE and IQ analyses)    
def make_tables2(datas, output_name):
    result_table = pd.DataFrame()
    covs_table = pd.DataFrame()
    all_table = pd.DataFrame()
    diff_table = pd.DataFrame()

    EN = datas[0].transpose()
    RF = datas[1].transpose()
    LGB = datas[2].transpose()
    SVM = datas[3].transpose()
    result_table['Elastic Net'] = EN[0].reset_index(drop=True)
    result_table['Random Forest'] = RF[0].reset_index(drop=True)
    result_table['Gradient Boosting'] = LGB[0].reset_index(drop=True)
    result_table['Support Vector Machine'] = SVM[0].reset_index(drop=True)
    result_table.index = ['smri','dmri','rs','nback','sst','mid']
    result_table.to_csv(output_name)

# load in results
results_base = pd.read_csv('drd_results_table_final.csv')
results_all = pd.read_csv('drd_results_table_final_full.csv')
results_site = pd.read_csv('drd_results_table_final_site.csv')
results_auc = pd.read_csv('drd_results_table_final_auc.csv')
results_valcheck4 = pd.read_csv('drd_results_table_final_4valcheck.csv')
results_valcheck2 = pd.read_csv('drd_results_table_final_2valcheck.csv')
results_valcheck1 = pd.read_csv('drd_results_table_final_1valcheck.csv')
results_rfe = pd.read_csv('drd_results_table_final_ridge_RFE.csv')
results_iq = pd.read_csv('IQ_results_table_final.csv')

#set labels
labels = ['Full-Sample','SMRI','DMRI','RS','NB','SST','MID']
labels_mri = ['Multimodal','SMRI','DMRI','RS','NB','SST','MID']


#########visualize primary analyses##########
#mri vars
results=results_all
x = np.r_[85:97, 0:73]
results = results.iloc[:,x]
data = results.filter(regex='_data')
covs = results.filter(regex='_covs')
alls  = results.filter(regex='_all')
results_EN = data.filter(regex='EN_')
results_RF = data.filter(regex='RF_')
results_LGB = data.filter(regex='LGB_')
results_SVM = data.filter(regex='SVM_')

covs_EN = covs.filter(regex='EN_')
covs_RF = covs.filter(regex='RF_')
covs_LGB = covs.filter(regex='LGB_')
covs_SVM = covs.filter(regex='SVM_')

all_EN = alls.filter(regex='EN_')
all_RF = alls.filter(regex='RF_')
all_LGB = alls.filter(regex='LGB_')
all_SVM= alls.filter(regex='SVM_')
print(all_EN)

x = np.arange(len(labels))  # the label locations
x = np.arange(len(list(results_EN.columns[0:7])))  # the label locations
width = 0.1  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - .25, zero_floor_diff(results_EN.iloc[0][0:7]), width, align='center',label='Elastic Net',color='LightGray')
rects2 = ax.bar(x - .15, zero_floor_diff(results_RF.iloc[0][0:7]), width, align='center',label='Random Forest',color='DarkGray')
rects3 = ax.bar(x - .05, zero_floor_diff(results_LGB.iloc[0][0:7]), width, align='center',label='Light GB',color='Gainsboro')
rects4 = ax.bar(x + .05, zero_floor_diff(results_SVM.iloc[0][0:7]), width, align='center',label='Support Vector',color='Gray')
ax.set_ylabel('R-square')
ax.set_xlabel('Sample')
ax.set_title('Prediction Using Brain Variables')
ax.set_xticks(x)
ax.set_xticklabels(labels_mri,)
ax.tick_params(axis='both',direction='out')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
ax.legend()
plt.legend(bbox_to_anchor=(1, 1))
plt.axhline(y=0, color='k', linestyle='-',linewidth=.5,)
plt.savefig('Data.tiff', dpi=500, bbox_inches='tight')

#psychosocial vars
results=results_base
data = results.filter(regex='_data')
covs = results.filter(regex='_covs')
alls  = results.filter(regex='_all')

covs_EN = covs.filter(regex='EN_')
covs_RF = covs.filter(regex='RF_')
covs_LGB = covs.filter(regex='LGB_')
covs_SVM = covs.filter(regex='SVM_')

all_EN = alls.filter(regex='EN_')
all_RF = alls.filter(regex='RF_')
all_LGB = alls.filter(regex='LGB_')
all_SVM= alls.filter(regex='SVM_')
print(all_EN)

x = np.arange(len(labels))  # the label locations
x = np.arange(len(list(covs_EN.columns[0:7])))  # the label locations
width = 0.1  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - .25, zero_floor_diff(covs_EN.iloc[0][0:7]), width, align='center',label='Elastic Net',color='LightGray')
rects2 = ax.bar(x - .15, zero_floor_diff(covs_RF.iloc[0][0:7]), width, align='center',label='Random Forest',color='DarkGray')
rects3 = ax.bar(x - .05, zero_floor_diff(covs_LGB.iloc[0][0:7]), width, align='center',label='Light GB',color='Gainsboro')
rects4 = ax.bar(x + .05, zero_floor_diff(covs_SVM.iloc[0][0:7]), width, align='center',label='Support Vector',color='Gray')
ax.set_ylabel('R-square')
ax.set_xlabel('Sample')
ax.set_title('Prediction Using Behavioral, Psychological, and Life History Variables')
ax.set_xticks(x)
ax.set_xticklabels(labels,)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
ax.tick_params(axis='both',direction='out')
ax.legend()
plt.legend(bbox_to_anchor=(1, 1))
plt.axhline(y=0, color='k', linestyle='-',linewidth=.5,)
plt.savefig('Covs.tiff', dpi=500, bbox_inches='tight')

x = np.arange(len(labels))  # the label locations
x = np.arange(len(list(all_EN.columns[1:7])))  # the label locations
width = 0.1  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - .25, zero_floor_diff(all_EN.iloc[0][1:7]), width, align='center',label='Elastic Net',color='LightGray')
rects2 = ax.bar(x - .15, zero_floor_diff(all_RF.iloc[0][1:7]), width, align='center',label='Random Forest',color='DarkGray')
rects3 = ax.bar(x - .05, zero_floor_diff(all_LGB.iloc[0][1:7]), width, align='center',label='Light GB',color='Gainsboro')
rects4 = ax.bar(x + .05, zero_floor_diff(all_SVM.iloc[0][1:7]), width, align='center',label='Support Vector',color='Gray')
ax.set_ylabel('R-square')
ax.set_xlabel('Sample')
ax.set_title('Prediction Using Brain and Psychosocial Variables')
ax.set_xticks(x)
ax.set_xticklabels(labels,)
ax.tick_params(axis='both',direction='out')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
ax.legend()
plt.legend(bbox_to_anchor=(1, 1))
plt.axhline(y=0, color='k', linestyle='-',linewidth=.5,)
plt.savefig('All.tiff', dpi=500, bbox_inches='tight')

diff_EN=matrix(zero_floor_only(all_EN.iloc[0][1:7]))-matrix(zero_floor_only(covs_EN.iloc[0][1:7]))
diff_RF=matrix(zero_floor_only(all_RF.iloc[0][1:7]))-matrix(zero_floor_only(covs_RF.iloc[0][1:7]))
diff_LGB=matrix(zero_floor_only(all_LGB.iloc[0][1:7]))-matrix(zero_floor_only(covs_LGB.iloc[0][1:7]))
diff_SVM=matrix(zero_floor_only(all_SVM.iloc[0][1:7]))-matrix(zero_floor_only(covs_SVM.iloc[0][1:7]))

x = np.arange(len(labels))  # the label locations
x = np.arange(len(list(all_EN.columns[1:7])))  # the label locations
width = 0.1  # the width of the bars

diff_EN=diff_EN[0,:].tolist()
diff_EN=diff_EN[0]

diff_RF=diff_RF[0,:].tolist()
diff_RF=diff_RF[0]

diff_LGB=diff_LGB[0,:].tolist()
diff_LGB=diff_LGB[0]

diff_SVM=diff_SVM[0,:].tolist()
diff_SVM=diff_SVM[0]

fig, ax = plt.subplots()
rects1 = ax.bar(x - .25, zero_floor_diff(diff_EN), width, align='center',label='Elastic Net',color='LightGray')
rects2 = ax.bar(x - .15, zero_floor_diff(diff_RF), width, align='center',label='Random Forest',color='DarkGray')
rects3 = ax.bar(x - .05, zero_floor_diff(diff_LGB), width, align='center',label='Light GB',color='Gainsboro')
rects4 = ax.bar(x + .05, zero_floor_diff(diff_SVM), width, align='center',label='Support Vector',color='Gray')
ax.set_ylabel('R-square')
ax.set_xlabel('Sample')
ax.set_title('Difference of Psychosocial + Brain vs Brain Only')
ax.set_xticks(x)
ax.set_xticklabels(labels,)
ax.tick_params(axis='both',direction='out')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
ax.legend()
plt.legend(bbox_to_anchor=(1, 1))
plt.axhline(y=0, color='k', linestyle='-',linewidth=.5,)
plt.savefig('Diff.tiff', dpi=500, bbox_inches='tight')

results = [results_EN, results_RF, results_LGB, results_SVM]
covs = [covs_EN, covs_RF, covs_LGB, covs_SVM]
alls = [all_EN, all_RF, all_LGB, all_SVM]
datas = [results, covs, alls]
output_name = ['results.csv', 'covs.csv', 'all.csv']

for i in range(3):
    make_tables(datas[i],output_name[i])

#############visualize supplementary analyses###########
#AUC Analyses
results=results_auc
data = results.filter(regex='_data')
covs = results.filter(regex='_covs')
alls  = results.filter(regex='_all')
results_EN = data.filter(regex='EN_')
results_RF = data.filter(regex='RF_')
results_LGB = data.filter(regex='LGB_')
results_SVM = data.filter(regex='SVM_')

covs_EN = covs.filter(regex='EN_')
covs_RF = covs.filter(regex='RF_')
covs_LGB = covs.filter(regex='LGB_')
covs_SVM = covs.filter(regex='SVM_')

all_EN = alls.filter(regex='EN_')
all_RF = alls.filter(regex='RF_')
all_LGB = alls.filter(regex='LGB_')
all_SVM= alls.filter(regex='SVM_')
print(all_EN)

labels = ['smri','dmri','rs','nb','sst','mid']#,'apriori','all','all (noqc)']

x = np.arange(len(labels))  # the label locations
x = np.arange(len(list(results_EN.columns[1:7])))  # the label locations
width = 0.1  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - .25, zero_floor_diff(results_EN.iloc[0][1:7]), width, align='center',label='Elastic Net',color='LightGray')
rects2 = ax.bar(x - .15, zero_floor_diff(results_RF.iloc[0][1:7]), width, align='center',label='Random Forest',color='DarkGray')
rects3 = ax.bar(x - .05, zero_floor_diff(results_LGB.iloc[0][1:7]), width, align='center',label='Light GB',color='Gainsboro')
rects4 = ax.bar(x + .05, zero_floor_diff(results_SVM.iloc[0][1:7]), width, align='center',label='Support Vector',color='Gray')
ax.set_ylabel('R-square')
ax.set_title('AUC - Prediction Using Brain Variables')
ax.set_xticks(x)
ax.set_xticklabels(labels,)
ax.tick_params(axis='both',direction='out')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
ax.legend()
plt.legend(bbox_to_anchor=(1, 1))
plt.axhline(y=0, color='k', linestyle='-',linewidth=.5,)
plt.savefig('Data_auc.tiff', dpi=500, bbox_inches='tight')

x = np.arange(len(labels))  # the label locations
x = np.arange(len(list(covs_EN.columns[0:7])))  # the label locations
width = 0.1  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - .25, zero_floor_diff(covs_EN.iloc[0][0:7]), width, align='center',label='Elastic Net',color='LightGray')
rects2 = ax.bar(x - .15, zero_floor_diff(covs_RF.iloc[0][0:7]), width, align='center',label='Random Forest',color='DarkGray')
rects3 = ax.bar(x - .05, zero_floor_diff(covs_LGB.iloc[0][0:7]), width, align='center',label='Light GB',color='Gainsboro')
rects4 = ax.bar(x + .05, zero_floor_diff(covs_SVM.iloc[0][0:7]), width, align='center',label='Support Vector',color='Gray')
ax.set_ylabel('R-square')
ax.set_title('AUC - Prediction Using Behavioral, Psychological, and Life History Variables')
ax.set_xticks(x)
ax.set_xticklabels(labels,)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
ax.tick_params(axis='both',direction='out')
ax.legend()
plt.legend(bbox_to_anchor=(1, 1))
plt.axhline(y=0, color='k', linestyle='-',linewidth=.5,)
plt.savefig('Covs_auc.tiff', dpi=500, bbox_inches='tight')

x = np.arange(len(labels))  # the label locations
x = np.arange(len(list(all_EN.columns[1:7])))  # the label locations
width = 0.1  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - .25, zero_floor_diff(all_EN.iloc[0][1:7]), width, align='center',label='Elastic Net',color='LightGray')
rects2 = ax.bar(x - .15, zero_floor_diff(all_RF.iloc[0][1:7]), width, align='center',label='Random Forest',color='DarkGray')
rects3 = ax.bar(x - .05, zero_floor_diff(all_LGB.iloc[0][1:7]), width, align='center',label='Light GB',color='Gainsboro')
rects4 = ax.bar(x + .05, zero_floor_diff(all_SVM.iloc[0][1:7]), width, align='center',label='Support Vector',color='Gray')
ax.set_ylabel('R-square')
ax.set_title('AUC - Prediction Using Brain and Psychosocial Variables')
ax.set_xticks(x)
ax.set_xticklabels(labels,)
ax.tick_params(axis='both',direction='out')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
ax.legend()
plt.legend(bbox_to_anchor=(1, 1))
plt.axhline(y=0, color='k', linestyle='-',linewidth=.5,)
plt.savefig('All_auc.tiff', dpi=500, bbox_inches='tight')

diff_EN=matrix(zero_floor_only(all_EN.iloc[0][1:7]))-matrix(zero_floor_only(covs_EN.iloc[0][1:7]))
diff_RF=matrix(zero_floor_only(all_RF.iloc[0][1:7]))-matrix(zero_floor_only(covs_RF.iloc[0][1:7]))
diff_LGB=matrix(zero_floor_only(all_LGB.iloc[0][1:7]))-matrix(zero_floor_only(covs_LGB.iloc[0][1:7]))
diff_SVM=matrix(zero_floor_only(all_SVM.iloc[0][1:7]))-matrix(zero_floor_only(covs_SVM.iloc[0][1:7]))

x = np.arange(len(labels))  # the label locations
x = np.arange(len(list(all_EN.columns[1:7])))  # the label locations
width = 0.1  # the width of the bars

diff_EN=diff_EN[0,:].tolist()
diff_EN=diff_EN[0]

diff_RF=diff_RF[0,:].tolist()
diff_RF=diff_RF[0]

diff_LGB=diff_LGB[0,:].tolist()
diff_LGB=diff_LGB[0]

diff_SVM=diff_SVM[0,:].tolist()
diff_SVM=diff_SVM[0]

fig, ax = plt.subplots()
rects1 = ax.bar(x - .25, zero_floor_diff(diff_EN), width, align='center',label='Elastic Net',color='LightGray')
rects2 = ax.bar(x - .15, zero_floor_diff(diff_RF), width, align='center',label='Random Forest',color='DarkGray')
rects3 = ax.bar(x - .05, zero_floor_diff(diff_LGB), width, align='center',label='Light GB',color='Gainsboro')
rects4 = ax.bar(x + .05, zero_floor_diff(diff_SVM), width, align='center',label='Support Vector',color='Gray')
ax.set_ylabel('R-square')
ax.set_title('AUC - Difference of Psychosocial + Brain vs Brain Only')
ax.set_xticks(x)
ax.set_xticklabels(labels,)
ax.tick_params(axis='both',direction='out')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
ax.legend()
plt.legend(bbox_to_anchor=(1, 1))
plt.axhline(y=0, color='k', linestyle='-',linewidth=.5,)
plt.savefig('Diff_auc.tiff', dpi=500, bbox_inches='tight')

results = [results_EN.iloc[:,np.r_[8,0:6]], results_RF.iloc[:,np.r_[8,0:6]], results_LGB.iloc[:,np.r_[8,0:6]], results_SVM.iloc[:,np.r_[8,0:6]]]
covs = [covs_EN.iloc[:,np.r_[8,0:6]], covs_RF.iloc[:,np.r_[8,0:6]], covs_LGB.iloc[:,np.r_[8,0:6]], covs_SVM.iloc[:,np.r_[8,0:6]]]
alls = [all_EN.iloc[:,np.r_[8,0:6]], all_RF.iloc[:,np.r_[8,0:6]], all_LGB.iloc[:,np.r_[8,0:6]], all_SVM.iloc[:,np.r_[8,0:6]]]
datas = [results, covs, alls]
output_name = ['results_auc.csv', 'covs_auc.csv', 'all_auc.csv']

for i in range(3):
    make_tables(datas[i],output_name[i])


#Exclusion of two inconsistencies on drd task
results=results_valcheck2
data = results.filter(regex='_data')
covs = results.filter(regex='_covs')
alls  = results.filter(regex='_all')
results_EN = data.filter(regex='EN_')
results_RF = data.filter(regex='RF_')
results_LGB = data.filter(regex='LGB_')
results_SVM = data.filter(regex='SVM_')

covs_EN = covs.filter(regex='EN_')
covs_RF = covs.filter(regex='RF_')
covs_LGB = covs.filter(regex='LGB_')
covs_SVM = covs.filter(regex='SVM_')

all_EN = alls.filter(regex='EN_')
all_RF = alls.filter(regex='RF_')
all_LGB = alls.filter(regex='LGB_')
all_SVM= alls.filter(regex='SVM_')
print(all_EN)

x = np.arange(len(labels))  # the label locations
x = np.arange(len(list(results_EN.columns[1:7])))  # the label locations
width = 0.1  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - .25, zero_floor_diff(results_EN.iloc[0][1:7]), width, align='center',label='Elastic Net',color='LightGray')
rects2 = ax.bar(x - .15, zero_floor_diff(results_RF.iloc[0][1:7]), width, align='center',label='Random Forest',color='DarkGray')
rects3 = ax.bar(x - .05, zero_floor_diff(results_LGB.iloc[0][1:7]), width, align='center',label='Light GB',color='Gainsboro')
rects4 = ax.bar(x + .05, zero_floor_diff(results_SVM.iloc[0][1:7]), width, align='center',label='Support Vector',color='Gray')
ax.set_ylabel('R-square')
ax.set_title('Valcheck2 - Prediction Using Brain Variables')
ax.set_xticks(x)
ax.set_xticklabels(labels,)
ax.tick_params(axis='both',direction='out')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
ax.legend()
plt.legend(bbox_to_anchor=(1, 1))
plt.axhline(y=0, color='k', linestyle='-',linewidth=.5,)
plt.savefig('Data_Valcheck2.tiff', dpi=500, bbox_inches='tight')

x = np.arange(len(labels))  # the label locations
x = np.arange(len(list(covs_EN.columns[0:7])))  # the label locations
width = 0.1  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - .25, zero_floor_diff(covs_EN.iloc[0][0:7]), width, align='center',label='Elastic Net',color='LightGray')
rects2 = ax.bar(x - .15, zero_floor_diff(covs_RF.iloc[0][0:7]), width, align='center',label='Random Forest',color='DarkGray')
rects3 = ax.bar(x - .05, zero_floor_diff(covs_LGB.iloc[0][0:7]), width, align='center',label='Light GB',color='Gainsboro')
rects4 = ax.bar(x + .05, zero_floor_diff(covs_SVM.iloc[0][0:7]), width, align='center',label='Support Vector',color='Gray')
ax.set_ylabel('R-square')
ax.set_title('Valcheck2 - Prediction Using Behavioral, Psychological, and Life History Variables')
ax.set_xticks(x)
ax.set_xticklabels(labels,)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
ax.tick_params(axis='both',direction='out')
ax.legend()
plt.legend(bbox_to_anchor=(1, 1))
plt.axhline(y=0, color='k', linestyle='-',linewidth=.5,)
plt.savefig('Covs_Valcheck2.tiff', dpi=500, bbox_inches='tight')

x = np.arange(len(labels))  # the label locations
x = np.arange(len(list(all_EN.columns[1:7])))  # the label locations
width = 0.1  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - .25, zero_floor_diff(all_EN.iloc[0][1:7]), width, align='center',label='Elastic Net',color='LightGray')
rects2 = ax.bar(x - .15, zero_floor_diff(all_RF.iloc[0][1:7]), width, align='center',label='Random Forest',color='DarkGray')
rects3 = ax.bar(x - .05, zero_floor_diff(all_LGB.iloc[0][1:7]), width, align='center',label='Light GB',color='Gainsboro')
rects4 = ax.bar(x + .05, zero_floor_diff(all_SVM.iloc[0][1:7]), width, align='center',label='Support Vector',color='Gray')
ax.set_ylabel('R-square')
ax.set_title('Valcheck2 - Prediction Using Brain and Psychosocial Variables')
ax.set_xticks(x)
ax.set_xticklabels(labels,)
ax.tick_params(axis='both',direction='out')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
ax.legend()
plt.legend(bbox_to_anchor=(1, 1))
plt.axhline(y=0, color='k', linestyle='-',linewidth=.5,)
plt.savefig('All_Valcheck2.tiff', dpi=500, bbox_inches='tight')

diff_EN=matrix(zero_floor_only(all_EN.iloc[0][1:7]))-matrix(zero_floor_only(covs_EN.iloc[0][1:7]))
diff_RF=matrix(zero_floor_only(all_RF.iloc[0][1:7]))-matrix(zero_floor_only(covs_RF.iloc[0][1:7]))
diff_LGB=matrix(zero_floor_only(all_LGB.iloc[0][1:7]))-matrix(zero_floor_only(covs_LGB.iloc[0][1:7]))
diff_SVM=matrix(zero_floor_only(all_SVM.iloc[0][1:7]))-matrix(zero_floor_only(covs_SVM.iloc[0][1:7]))

x = np.arange(len(labels))  # the label locations
x = np.arange(len(list(all_EN.columns[1:7])))  # the label locations
width = 0.1  # the width of the bars

diff_EN=diff_EN[0,:].tolist()
diff_EN=diff_EN[0]

diff_RF=diff_RF[0,:].tolist()
diff_RF=diff_RF[0]

diff_LGB=diff_LGB[0,:].tolist()
diff_LGB=diff_LGB[0]

diff_SVM=diff_SVM[0,:].tolist()
diff_SVM=diff_SVM[0]

fig, ax = plt.subplots()
rects1 = ax.bar(x - .25, zero_floor_diff(diff_EN), width, align='center',label='Elastic Net',color='LightGray')
rects2 = ax.bar(x - .15, zero_floor_diff(diff_RF), width, align='center',label='Random Forest',color='DarkGray')
rects3 = ax.bar(x - .05, zero_floor_diff(diff_LGB), width, align='center',label='Light GB',color='Gainsboro')
rects4 = ax.bar(x + .05, zero_floor_diff(diff_SVM), width, align='center',label='Support Vector',color='Gray')
ax.set_ylabel('R-square')
ax.set_title('Valcheck2 - Difference of Psychosocial + Brain vs Brain Only')
ax.set_xticks(x)
ax.set_xticklabels(labels,)
ax.tick_params(axis='both',direction='out')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
ax.legend()
plt.legend(bbox_to_anchor=(1, 1))
plt.axhline(y=0, color='k', linestyle='-',linewidth=.5,)
plt.savefig('Diff_Valcheck2.tiff', dpi=500, bbox_inches='tight')

results = [results_EN.iloc[:,np.r_[8,0:6]], results_RF.iloc[:,np.r_[8,0:6]], results_LGB.iloc[:,np.r_[8,0:6]], results_SVM.iloc[:,np.r_[8,0:6]]]
covs = [covs_EN.iloc[:,np.r_[8,0:6]], covs_RF.iloc[:,np.r_[8,0:6]], covs_LGB.iloc[:,np.r_[8,0:6]], covs_SVM.iloc[:,np.r_[8,0:6]]]
alls = [all_EN.iloc[:,np.r_[8,0:6]], all_RF.iloc[:,np.r_[8,0:6]], all_LGB.iloc[:,np.r_[8,0:6]], all_SVM.iloc[:,np.r_[8,0:6]]]
datas = [results, covs, alls]
output_name = ['results_valcheck2.csv', 'covs_valcheck2.csv', 'all_valcheck2.csv']

for i in range(3):
    make_tables(datas[i],output_name[i])

#Exclusion of one inconsistencies on drd task
results=results_valcheck1
data = results.filter(regex='_data')
covs = results.filter(regex='_covs')
alls  = results.filter(regex='_all')
results_EN = data.filter(regex='EN_')
results_RF = data.filter(regex='RF_')
results_LGB = data.filter(regex='LGB_')
results_SVM = data.filter(regex='SVM_')

covs_EN = covs.filter(regex='EN_')
covs_RF = covs.filter(regex='RF_')
covs_LGB = covs.filter(regex='LGB_')
covs_SVM = covs.filter(regex='SVM_')

all_EN = alls.filter(regex='EN_')
all_RF = alls.filter(regex='RF_')
all_LGB = alls.filter(regex='LGB_')
all_SVM= alls.filter(regex='SVM_')
print(all_EN)

x = np.arange(len(labels))  # the label locations
x = np.arange(len(list(results_EN.columns[1:7])))  # the label locations
width = 0.1  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - .25, zero_floor_diff(results_EN.iloc[0][1:7]), width, align='center',label='Elastic Net',color='LightGray')
rects2 = ax.bar(x - .15, zero_floor_diff(results_RF.iloc[0][1:7]), width, align='center',label='Random Forest',color='DarkGray')
rects3 = ax.bar(x - .05, zero_floor_diff(results_LGB.iloc[0][1:7]), width, align='center',label='Light GB',color='Gainsboro')
rects4 = ax.bar(x + .05, zero_floor_diff(results_SVM.iloc[0][1:7]), width, align='center',label='Support Vector',color='Gray')
ax.set_ylabel('R-square')
ax.set_title('Valcheck1 - Prediction Using Brain Variables')
ax.set_xticks(x)
ax.set_xticklabels(labels,)
ax.tick_params(axis='both',direction='out')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
ax.legend()
plt.legend(bbox_to_anchor=(1, 1))
plt.axhline(y=0, color='k', linestyle='-',linewidth=.5,)
plt.savefig('Data_Valcheck1.tiff', dpi=500, bbox_inches='tight')

x = np.arange(len(labels))  # the label locations
x = np.arange(len(list(covs_EN.columns[0:7])))  # the label locations
width = 0.1  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - .25, zero_floor_diff(covs_EN.iloc[0][0:7]), width, align='center',label='Elastic Net',color='LightGray')
rects2 = ax.bar(x - .15, zero_floor_diff(covs_RF.iloc[0][0:7]), width, align='center',label='Random Forest',color='DarkGray')
rects3 = ax.bar(x - .05, zero_floor_diff(covs_LGB.iloc[0][0:7]), width, align='center',label='Light GB',color='Gainsboro')
rects4 = ax.bar(x + .05, zero_floor_diff(covs_SVM.iloc[0][0:7]), width, align='center',label='Support Vector',color='Gray')
ax.set_ylabel('R-square')
ax.set_title('Valcheck1 - Prediction Using Behavioral, Psychological, and Life History Variables')
ax.set_xticks(x)
ax.set_xticklabels(labels,)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
ax.tick_params(axis='both',direction='out')
ax.legend()
plt.legend(bbox_to_anchor=(1, 1))
plt.axhline(y=0, color='k', linestyle='-',linewidth=.5,)
plt.savefig('Covs_Valcheck1.tiff', dpi=500, bbox_inches='tight')

x = np.arange(len(labels))  # the label locations
x = np.arange(len(list(all_EN.columns[1:7])))  # the label locations
width = 0.1  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - .25, zero_floor_diff(all_EN.iloc[0][1:7]), width, align='center',label='Elastic Net',color='LightGray')
rects2 = ax.bar(x - .15, zero_floor_diff(all_RF.iloc[0][1:7]), width, align='center',label='Random Forest',color='DarkGray')
rects3 = ax.bar(x - .05, zero_floor_diff(all_LGB.iloc[0][1:7]), width, align='center',label='Light GB',color='Gainsboro')
rects4 = ax.bar(x + .05, zero_floor_diff(all_SVM.iloc[0][1:7]), width, align='center',label='Support Vector',color='Gray')
ax.set_ylabel('R-square')
ax.set_title('Valcheck1 - Prediction Using Brain and Psychosocial Variables')
ax.set_xticks(x)
ax.set_xticklabels(labels,)
ax.tick_params(axis='both',direction='out')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
ax.legend()
plt.legend(bbox_to_anchor=(1, 1))
plt.axhline(y=0, color='k', linestyle='-',linewidth=.5,)
plt.savefig('All_Valcheck1.tiff', dpi=500, bbox_inches='tight')

from numpy import matrix
diff_EN=matrix(zero_floor_only(all_EN.iloc[0][1:7]))-matrix(zero_floor_only(covs_EN.iloc[0][1:7]))
diff_RF=matrix(zero_floor_only(all_RF.iloc[0][1:7]))-matrix(zero_floor_only(covs_RF.iloc[0][1:7]))
diff_LGB=matrix(zero_floor_only(all_LGB.iloc[0][1:7]))-matrix(zero_floor_only(covs_LGB.iloc[0][1:7]))
diff_SVM=matrix(zero_floor_only(all_SVM.iloc[0][1:7]))-matrix(zero_floor_only(covs_SVM.iloc[0][1:7]))

x = np.arange(len(labels))  # the label locations
x = np.arange(len(list(all_EN.columns[1:7])))  # the label locations
width = 0.1  # the width of the bars

diff_EN=diff_EN[0,:].tolist()
diff_EN=diff_EN[0]

diff_RF=diff_RF[0,:].tolist()
diff_RF=diff_RF[0]

diff_LGB=diff_LGB[0,:].tolist()
diff_LGB=diff_LGB[0]

diff_SVM=diff_SVM[0,:].tolist()
diff_SVM=diff_SVM[0]

fig, ax = plt.subplots()
rects1 = ax.bar(x - .25, zero_floor_diff(diff_EN), width, align='center',label='Elastic Net',color='LightGray')
rects2 = ax.bar(x - .15, zero_floor_diff(diff_RF), width, align='center',label='Random Forest',color='DarkGray')
rects3 = ax.bar(x - .05, zero_floor_diff(diff_LGB), width, align='center',label='Light GB',color='Gainsboro')
rects4 = ax.bar(x + .05, zero_floor_diff(diff_SVM), width, align='center',label='Support Vector',color='Gray')
ax.set_ylabel('R-square')
ax.set_title('Valcheck1 - Difference of Psychosocial + Brain vs Brain Only')
ax.set_xticks(x)
ax.set_xticklabels(labels,)
ax.tick_params(axis='both',direction='out')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
ax.legend()
plt.legend(bbox_to_anchor=(1, 1))
plt.axhline(y=0, color='k', linestyle='-',linewidth=.5,)
plt.savefig('Diff_Valcheck1.tiff', dpi=500, bbox_inches='tight')

results = [results_EN.iloc[:,np.r_[8,0:6]], results_RF.iloc[:,np.r_[8,0:6]], results_LGB.iloc[:,np.r_[8,0:6]], results_SVM.iloc[:,np.r_[8,0:6]]]
covs = [covs_EN.iloc[:,np.r_[8,0:6]], covs_RF.iloc[:,np.r_[8,0:6]], covs_LGB.iloc[:,np.r_[8,0:6]], covs_SVM.iloc[:,np.r_[8,0:6]]]
alls = [all_EN.iloc[:,np.r_[8,0:6]], all_RF.iloc[:,np.r_[8,0:6]], all_LGB.iloc[:,np.r_[8,0:6]], all_SVM.iloc[:,np.r_[8,0:6]]]
datas = [results, covs, alls]
output_name = ['results_valcheck1.csv', 'covs_valcheck1.csv', 'all_valcheck1.csv']

for i in range(3):
    make_tables(datas[i],output_name[i])

#Exclusion of four inconsistencies on drd task
results=results_valcheck4
data = results.filter(regex='_data')
covs = results.filter(regex='_covs')
alls  = results.filter(regex='_all')
results_EN = data.filter(regex='EN_')
results_RF = data.filter(regex='RF_')
results_LGB = data.filter(regex='LGB_')
results_SVM = data.filter(regex='SVM_')

covs_EN = covs.filter(regex='EN_')
covs_RF = covs.filter(regex='RF_')
covs_LGB = covs.filter(regex='LGB_')
covs_SVM = covs.filter(regex='SVM_')

all_EN = alls.filter(regex='EN_')
all_RF = alls.filter(regex='RF_')
all_LGB = alls.filter(regex='LGB_')
all_SVM= alls.filter(regex='SVM_')
print(all_EN)

x = np.arange(len(labels))  # the label locations
x = np.arange(len(list(results_EN.columns[1:7])))  # the label locations
width = 0.1  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - .25, zero_floor_diff(results_EN.iloc[0][1:7]), width, align='center',label='Elastic Net',color='LightGray')
rects2 = ax.bar(x - .15, zero_floor_diff(results_RF.iloc[0][1:7]), width, align='center',label='Random Forest',color='DarkGray')
rects3 = ax.bar(x - .05, zero_floor_diff(results_LGB.iloc[0][1:7]), width, align='center',label='Light GB',color='Gainsboro')
rects4 = ax.bar(x + .05, zero_floor_diff(results_SVM.iloc[0][1:7]), width, align='center',label='Support Vector',color='Gray')
ax.set_ylabel('R-square')
ax.set_title('Valcheck4 - Prediction Using Brain Variables')
ax.set_xticks(x)
ax.set_xticklabels(labels,)
ax.tick_params(axis='both',direction='out')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
ax.legend()
plt.legend(bbox_to_anchor=(1, 1))
plt.axhline(y=0, color='k', linestyle='-',linewidth=.5,)
plt.savefig('Data_Valcheck4.tiff', dpi=500, bbox_inches='tight')

x = np.arange(len(labels))  # the label locations
x = np.arange(len(list(covs_EN.columns[0:7])))  # the label locations
width = 0.1  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - .25, zero_floor_diff(covs_EN.iloc[0][0:7]), width, align='center',label='Elastic Net',color='LightGray')
rects2 = ax.bar(x - .15, zero_floor_diff(covs_RF.iloc[0][0:7]), width, align='center',label='Random Forest',color='DarkGray')
rects3 = ax.bar(x - .05, zero_floor_diff(covs_LGB.iloc[0][0:7]), width, align='center',label='Light GB',color='Gainsboro')
rects4 = ax.bar(x + .05, zero_floor_diff(covs_SVM.iloc[0][0:7]), width, align='center',label='Support Vector',color='Gray')
ax.set_ylabel('R-square')
ax.set_title('Valcheck4 - Prediction Using Behavioral, Psychological, and Life History Variables')
ax.set_xticks(x)
ax.set_xticklabels(labels,)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
ax.tick_params(axis='both',direction='out')
ax.legend()
plt.legend(bbox_to_anchor=(1, 1))
plt.axhline(y=0, color='k', linestyle='-',linewidth=.5,)
plt.savefig('Covs_Valcheck4.tiff', dpi=500, bbox_inches='tight')

x = np.arange(len(labels))  # the label locations
x = np.arange(len(list(all_EN.columns[1:7])))  # the label locations
width = 0.1  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - .25, zero_floor_diff(all_EN.iloc[0][1:7]), width, align='center',label='Elastic Net',color='LightGray')
rects2 = ax.bar(x - .15, zero_floor_diff(all_RF.iloc[0][1:7]), width, align='center',label='Random Forest',color='DarkGray')
rects3 = ax.bar(x - .05, zero_floor_diff(all_LGB.iloc[0][1:7]), width, align='center',label='Light GB',color='Gainsboro')
rects4 = ax.bar(x + .05, zero_floor_diff(all_SVM.iloc[0][1:7]), width, align='center',label='Support Vector',color='Gray')
ax.set_ylabel('R-square')
ax.set_title('Valcheck4 - Prediction Using Brain and Psychosocial Variables')
ax.set_xticks(x)
ax.set_xticklabels(labels,)
ax.tick_params(axis='both',direction='out')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
ax.legend()
plt.legend(bbox_to_anchor=(1, 1))
plt.axhline(y=0, color='k', linestyle='-',linewidth=.5,)
plt.savefig('All_Valcheck4.tiff', dpi=500, bbox_inches='tight')

diff_EN=matrix(zero_floor_only(all_EN.iloc[0][1:7]))-matrix(zero_floor_only(covs_EN.iloc[0][1:7]))
diff_RF=matrix(zero_floor_only(all_RF.iloc[0][1:7]))-matrix(zero_floor_only(covs_RF.iloc[0][1:7]))
diff_LGB=matrix(zero_floor_only(all_LGB.iloc[0][1:7]))-matrix(zero_floor_only(covs_LGB.iloc[0][1:7]))
diff_SVM=matrix(zero_floor_only(all_SVM.iloc[0][1:7]))-matrix(zero_floor_only(covs_SVM.iloc[0][1:7]))

x = np.arange(len(labels))  # the label locations
x = np.arange(len(list(all_EN.columns[1:7])))  # the label locations
width = 0.1  # the width of the bars

diff_EN=diff_EN[0,:].tolist()
diff_EN=diff_EN[0]

diff_RF=diff_RF[0,:].tolist()
diff_RF=diff_RF[0]

diff_LGB=diff_LGB[0,:].tolist()
diff_LGB=diff_LGB[0]

diff_SVM=diff_SVM[0,:].tolist()
diff_SVM=diff_SVM[0]

fig, ax = plt.subplots()
rects1 = ax.bar(x - .25, zero_floor_diff(diff_EN), width, align='center',label='Elastic Net',color='LightGray')
rects2 = ax.bar(x - .15, zero_floor_diff(diff_RF), width, align='center',label='Random Forest',color='DarkGray')
rects3 = ax.bar(x - .05, zero_floor_diff(diff_LGB), width, align='center',label='Light GB',color='Gainsboro')
rects4 = ax.bar(x + .05, zero_floor_diff(diff_SVM), width, align='center',label='Support Vector',color='Gray')
ax.set_ylabel('R-square')
ax.set_title('Valcheck4 - Difference of Psychosocial + Brain vs Brain Only')
ax.set_xticks(x)
ax.set_xticklabels(labels,)
ax.tick_params(axis='both',direction='out')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
ax.legend()
plt.legend(bbox_to_anchor=(1, 1))
plt.axhline(y=0, color='k', linestyle='-',linewidth=.5,)
plt.savefig('Diff_Valcheck4.tiff', dpi=500, bbox_inches='tight')

results = [results_EN.iloc[:,np.r_[8,0:6]], results_RF.iloc[:,np.r_[8,0:6]], results_LGB.iloc[:,np.r_[8,0:6]], results_SVM.iloc[:,np.r_[8,0:6]]]
covs = [covs_EN.iloc[:,np.r_[8,0:6]], covs_RF.iloc[:,np.r_[8,0:6]], covs_LGB.iloc[:,np.r_[8,0:6]], covs_SVM.iloc[:,np.r_[8,0:6]]]
alls = [all_EN.iloc[:,np.r_[8,0:6]], all_RF.iloc[:,np.r_[8,0:6]], all_LGB.iloc[:,np.r_[8,0:6]], all_SVM.iloc[:,np.r_[8,0:6]]]
datas = [results, covs, alls]
output_name = ['results_valcheck4.csv', 'covs_valcheck4.csv', 'all_valcheck4.csv']

for i in range(3):
    make_tables(datas[i],output_name[i])

#Site as Grouping Variable instead of Family
results=results_site
data = results.filter(regex='_data')
covs = results.filter(regex='_covs')
alls  = results.filter(regex='_all')
results_EN = data.filter(regex='EN_')
results_RF = data.filter(regex='RF_')
results_LGB = data.filter(regex='LGB_')
results_SVM = data.filter(regex='SVM_')

covs_EN = covs.filter(regex='EN_')
covs_RF = covs.filter(regex='RF_')
covs_LGB = covs.filter(regex='LGB_')
covs_SVM = covs.filter(regex='SVM_')

all_EN = alls.filter(regex='EN_')
all_RF = alls.filter(regex='RF_')
all_LGB = alls.filter(regex='LGB_')
all_SVM= alls.filter(regex='SVM_')
print(all_EN)

x = np.arange(len(labels))  # the label locations
x = np.arange(len(list(results_EN.columns[1:7])))  # the label locations
width = 0.1  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - .25, zero_floor_diff(results_EN.iloc[0][1:7]), width, align='center',label='Elastic Net',color='LightGray')
rects2 = ax.bar(x - .15, zero_floor_diff(results_RF.iloc[0][1:7]), width, align='center',label='Random Forest',color='DarkGray')
rects3 = ax.bar(x - .05, zero_floor_diff(results_LGB.iloc[0][1:7]), width, align='center',label='Light GB',color='Gainsboro')
rects4 = ax.bar(x + .05, zero_floor_diff(results_SVM.iloc[0][1:7]), width, align='center',label='Support Vector',color='Gray')
ax.set_ylabel('R-square')
ax.set_title('Site Stratified - Prediction Using Brain Variables')
ax.set_xticks(x)
ax.set_xticklabels(labels,)
ax.tick_params(axis='both',direction='out')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
ax.legend()
plt.legend(bbox_to_anchor=(1, 1))
plt.axhline(y=0, color='k', linestyle='-',linewidth=.5,)
plt.savefig('Data_site.tiff', dpi=500, bbox_inches='tight')

x = np.arange(len(labels))  # the label locations
x = np.arange(len(list(covs_EN.columns[0:7])))  # the label locations
width = 0.1  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - .25, zero_floor_diff(covs_EN.iloc[0][0:7]), width, align='center',label='Elastic Net',color='LightGray')
rects2 = ax.bar(x - .15, zero_floor_diff(covs_RF.iloc[0][0:7]), width, align='center',label='Random Forest',color='DarkGray')
rects3 = ax.bar(x - .05, zero_floor_diff(covs_LGB.iloc[0][0:7]), width, align='center',label='Light GB',color='Gainsboro')
rects4 = ax.bar(x + .05, zero_floor_diff(covs_SVM.iloc[0][0:7]), width, align='center',label='Support Vector',color='Gray')
ax.set_ylabel('R-square')
ax.set_title('Site Stratified - Prediction Using Behavioral, Psychological, and Life History Variables')
ax.set_xticks(x)
ax.set_xticklabels(labels,)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
ax.tick_params(axis='both',direction='out')
ax.legend()
plt.legend(bbox_to_anchor=(1, 1))
plt.axhline(y=0, color='k', linestyle='-',linewidth=.5,)
plt.savefig('Covs_site.tiff', dpi=500, bbox_inches='tight')

x = np.arange(len(labels))  # the label locations
x = np.arange(len(list(all_EN.columns[1:7])))  # the label locations
width = 0.1  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - .25, zero_floor_diff(all_EN.iloc[0][1:7]), width, align='center',label='Elastic Net',color='LightGray')
rects2 = ax.bar(x - .15, zero_floor_diff(all_RF.iloc[0][1:7]), width, align='center',label='Random Forest',color='DarkGray')
rects3 = ax.bar(x - .05, zero_floor_diff(all_LGB.iloc[0][1:7]), width, align='center',label='Light GB',color='Gainsboro')
rects4 = ax.bar(x + .05, zero_floor_diff(all_SVM.iloc[0][1:7]), width, align='center',label='Support Vector',color='Gray')
ax.set_ylabel('R-square')
ax.set_title('Site Stratified - Prediction Using Brain and Psychosocial Variables')
ax.set_xticks(x)
ax.set_xticklabels(labels,)
ax.tick_params(axis='both',direction='out')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
ax.legend()
plt.legend(bbox_to_anchor=(1, 1))
plt.axhline(y=0, color='k', linestyle='-',linewidth=.5,)
plt.savefig('All_site.tiff', dpi=500, bbox_inches='tight')

diff_EN=matrix(zero_floor_only(all_EN.iloc[0][1:7]))-matrix(zero_floor_only(covs_EN.iloc[0][1:7]))
diff_RF=matrix(zero_floor_only(all_RF.iloc[0][1:7]))-matrix(zero_floor_only(covs_RF.iloc[0][1:7]))
diff_LGB=matrix(zero_floor_only(all_LGB.iloc[0][1:7]))-matrix(zero_floor_only(covs_LGB.iloc[0][1:7]))
diff_SVM=matrix(zero_floor_only(all_SVM.iloc[0][1:7]))-matrix(zero_floor_only(covs_SVM.iloc[0][1:7]))

x = np.arange(len(labels))  # the label locations
x = np.arange(len(list(all_EN.columns[1:7])))  # the label locations
width = 0.1  # the width of the bars

diff_EN=diff_EN[0,:].tolist()
diff_EN=diff_EN[0]

diff_RF=diff_RF[0,:].tolist()
diff_RF=diff_RF[0]

diff_LGB=diff_LGB[0,:].tolist()
diff_LGB=diff_LGB[0]

diff_SVM=diff_SVM[0,:].tolist()
diff_SVM=diff_SVM[0]

fig, ax = plt.subplots()
rects1 = ax.bar(x - .25, zero_floor_diff(diff_EN), width, align='center',label='Elastic Net',color='LightGray')
rects2 = ax.bar(x - .15, zero_floor_diff(diff_RF), width, align='center',label='Random Forest',color='DarkGray')
rects3 = ax.bar(x - .05, zero_floor_diff(diff_LGB), width, align='center',label='Light GB',color='Gainsboro')
rects4 = ax.bar(x + .05, zero_floor_diff(diff_SVM), width, align='center',label='Support Vector',color='Gray')
ax.set_ylabel('R-square')
ax.set_title('Site Stratified - Difference of Psychosocial + Brain vs Brain Only')
ax.set_xticks(x)
ax.set_xticklabels(labels,)
ax.tick_params(axis='both',direction='out')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
ax.legend()
plt.legend(bbox_to_anchor=(1, 1))
plt.axhline(y=0, color='k', linestyle='-',linewidth=.5,)
plt.savefig('Diff_site.tiff', dpi=500, bbox_inches='tight')

results = [results_EN.iloc[:,np.r_[8,0:6]], results_RF.iloc[:,np.r_[8,0:6]], results_LGB.iloc[:,np.r_[8,0:6]], results_SVM.iloc[:,np.r_[8,0:6]]]
covs = [covs_EN.iloc[:,np.r_[8,0:6]], covs_RF.iloc[:,np.r_[8,0:6]], covs_LGB.iloc[:,np.r_[8,0:6]], covs_SVM.iloc[:,np.r_[8,0:6]]]
alls = [all_EN.iloc[:,np.r_[8,0:6]], all_RF.iloc[:,np.r_[8,0:6]], all_LGB.iloc[:,np.r_[8,0:6]], all_SVM.iloc[:,np.r_[8,0:6]]]
datas = [results, covs, alls]
output_name = ['results_site.csv', 'covs_site.csv', 'all_site.csv']

for i in range(3):
    make_tables(datas[i],output_name[i])

#Recursive Feature Elimination
results=results_rfe
data = results.filter(regex='_data')
covs = results.filter(regex='_covs')
alls  = results.filter(regex='_all')
results_EN = data.filter(regex='EN_')
results_RF = data.filter(regex='RF_')
results_LGB = data.filter(regex='LGB_')
results_SVM = data.filter(regex='SVM_')

covs_EN = covs.filter(regex='EN_')
covs_RF = covs.filter(regex='RF_')
covs_LGB = covs.filter(regex='LGB_')
covs_SVM = covs.filter(regex='SVM_')

all_EN = alls.filter(regex='EN_')
all_RF = alls.filter(regex='RF_')
all_LGB = alls.filter(regex='LGB_')
all_SVM= alls.filter(regex='SVM_')
print(all_EN)

x = np.arange(len(labels))  # the label locations
x = np.arange(len(list(results_EN.columns[1:7])))  # the label locations
width = 0.1  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - .25, zero_floor_diff(results_EN.iloc[0][1:7]), width, align='center',label='Elastic Net',color='LightGray')
rects2 = ax.bar(x - .15, zero_floor_diff(results_RF.iloc[0][1:7]), width, align='center',label='Random Forest',color='DarkGray')
rects3 = ax.bar(x - .05, zero_floor_diff(results_LGB.iloc[0][1:7]), width, align='center',label='Light GB',color='Gainsboro')
rects4 = ax.bar(x + .05, zero_floor_diff(results_SVM.iloc[0][1:7]), width, align='center',label='Support Vector',color='Gray')
ax.set_ylabel('R-square')
ax.set_title('RFE - Prediction Using Brain Variables')
ax.set_xticks(x)
ax.set_xticklabels(labels,)
ax.tick_params(axis='both',direction='out')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
ax.legend()
plt.legend(bbox_to_anchor=(1, 1))
plt.axhline(y=0, color='k', linestyle='-',linewidth=.5,)
plt.savefig('Data_rfe.tiff', dpi=500, bbox_inches='tight')

results = [results_EN.iloc[:,0:6], results_RF.iloc[:,0:6], results_LGB.iloc[:,0:6], results_SVM.iloc[:,0:6]]
output_name = 'results_rfe.csv'

make_tables2(results,output_name)

#IQ
results=results_iq
data = results.filter(regex='_data')
results_EN = data.filter(regex='EN_')
results_RF = data.filter(regex='RF_')
results_LGB = data.filter(regex='LGB_')
results_SVM = data.filter(regex='SVM_')

labels = ['smri','dmri','rs','nb','sst','mid']#,'apriori','all','all (noqc)']

x = np.arange(len(labels))  # the label locations
x = np.arange(len(list(results_EN.columns[0:6])))  # the label locations
width = 0.1  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - .25, zero_floor_diff(results_EN.iloc[0][0:6]), width, align='center',label='Elastic Net',color='LightGray')
rects2 = ax.bar(x - .15, zero_floor_diff(results_RF.iloc[0][0:6]), width, align='center',label='Random Forest',color='DarkGray')
rects3 = ax.bar(x - .05, zero_floor_diff(results_LGB.iloc[0][0:6]), width, align='center',label='Light GB',color='Gainsboro')
rects4 = ax.bar(x + .05, zero_floor_diff(results_SVM.iloc[0][0:6]), width, align='center',label='Support Vector',color='Gray')
ax.set_ylabel('R-square')
ax.set_title('Prediction Using Brain Variables')
ax.set_xticks(x)
ax.set_xticklabels(labels,)
ax.tick_params(axis='both',direction='out')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
ax.legend()
plt.legend(bbox_to_anchor=(1, 1))
plt.axhline(y=0, color='k', linestyle='-',linewidth=.5,)
plt.savefig('Data_iq.tiff', dpi=500, bbox_inches='tight')


results = [results_EN.iloc[:,0:6], results_RF.iloc[:,0:6], results_LGB.iloc[:,0:6], results_SVM.iloc[:,0:6]]
output_name = 'results_iq.csv'

make_tables2(results,output_name)