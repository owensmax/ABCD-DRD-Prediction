#set working directory
import os
os.chdir('/home/max/Documents/DRD/final_analyses_for_github_with_data')

#import libraries
from loading_final import get_setup_ML, get_cat_vars, load_data
from BPt import * #v 1.3.4
import pandas as pd #v 1.1.5
import scipy.stats #v 1.5.0
import statsmodels as sm #v 0.11.1

#################DEFINE FUNCTIONS##############
def evaluate(pipeline, target='mean_indif', scope='covars'):
    
    ps = Problem_Spec(scorer='r2',
                      scope=scope,
                      target=target,
                      random_state = 1001,
                      n_jobs=8)
    
    return ML.Evaluate(model_pipeline=pipeline,
                       problem_spec=ps,
                       splits=5,
                       n_repeats=1,
                       feat_importances='base')
    
def test(pipeline, target='mean_indif', scope='covars'):
    
    ps = Problem_Spec(scorer='r2',
                      scope=scope,
                      target=target,
                      random_state = 1001,
                      n_jobs=8)
    
    return ML.Test(model_pipeline=pipeline,
                       problem_spec=ps,
                       feat_importances='base')
    
def test_shap(pipeline, target='mean_indif', scope='covars'):
    
    ps = Problem_Spec(scorer='r2',
                      scope=scope,
                      target=target,
                      random_state = 1001,
                      n_jobs=8)
    
    fi = Feat_Importance('shap', scorer='default', 
                             shap_params='default', 
                             n_perm=10, inverse_global=False, 
                             inverse_local=False)
    
    return ML.Test(model_pipeline=pipeline,
                       problem_spec=ps,
                       feat_importances=fi)

def test_perm(pipeline, target='mean_indif', scope='covars'):
    
    ps = Problem_Spec(scorer='r2',
                      scope=scope,
                      target=target,
                      random_state = 1001,
                      n_jobs=8)
    
    fi = Feat_Importance('perm', scorer='default', 
                             shap_params='default', 
                             n_perm=10, inverse_global=False, 
                             inverse_local=False)
    
    return ML.Test(model_pipeline=pipeline,
                       problem_spec=ps,
                       feat_importances=fi)

def test_rfe(pipeline, target='mean_indif', scope='covars'):
    
    ps = Problem_Spec(scorer='r2',
                      scope=scope,
                      target=target,
                      random_state = 1001,
                      n_jobs=8)
    
    return ML.Test(model_pipeline=pipeline,
                       problem_spec=ps,
                       feat_importances='base')

##################DEFINE ML PIPELIENES################
# Scalers
scalers = [Scaler('winsorize', scope='float'),
           Scaler('standard', scope='float')]
robust_scaler = Scaler('robust', scope='float')

# Param searches
alt_search = Param_Search(search_type='DiscreteOnePlusOne',
                          n_iter= 60)
# Imputers
imputers = [Imputer('mean', scope='float'), 
            Imputer('median', scope='cat')]

# Feat Selectors
linear_rfe = Feat_Selector('rfe', base_model=Model('linear'))
ridge_rfe = Feat_Selector('rfe', base_model=Model('ridge', params=1))
rf_rfe = Feat_Selector('rfe', base_model=Model('random forest', params=1))

# Transformers
cat_vars = get_cat_vars()
scope = Duplicate(cat_vars)
ohe = Transformer('one hot encoder', scope=scope)


EN = Model_Pipeline(imputers=imputers,
                          scalers=robust_scaler,
                          transformers=ohe,
                          model = Model('elastic net regressor', params=1),
                          param_search = alt_search)
RF = Model_Pipeline(imputers=imputers,
                          scalers=robust_scaler,
                          transformers=ohe,
                          model = Model('random forest regressor', params=1),
                          param_search = alt_search)

LGB = Model_Pipeline(imputers=imputers,
                          scalers=robust_scaler,
                          transformers=ohe,
                          model = Model('light gbm regressor', params=1),
                          param_search = alt_search)

SVM = Model_Pipeline(imputers=imputers,
                          scalers=robust_scaler,
                          transformers=ohe,
                          model = Model('svm regressor', params=1),
                          param_search = alt_search)

pipelines = [EN,RF,LGB,SVM]
pipe_names = ['EN','RF','LGB','SVM']

modalities = ['smri','dmri','rs','nback','sst','mid','apriori','all','all_noqc']

###########SET TEST SUBS FOR ALL ANALYSES###########
with open('final_test_subs.txt') as f:
    test_subs = f.read().splitlines()

####################PRIMARY ANALYSIS############################
result_store={}
result_table=pd.DataFrame(index=['r2','micro_sd','macro_sd'])
for modality in modalities:
    for pipeline,pipe_name in zip(pipelines,pipe_names):

        ML = get_setup_ML(get=modality,test_subjects=test_subs)

        covar_results = evaluate(pipeline, 'mean_indif', 'covars')
        data_results = evaluate(pipeline, 'mean_indif', 'data')
        all_results = evaluate(pipeline, 'mean_indif', 'all')

        result_store['{}_{}_covs'.format(pipe_name,modality)]=covar_results
        result_table['{}_{}_covs'.format(pipe_name,modality)]=covar_results['summary_scores'][0]

        result_store['{}_{}_data'.format(pipe_name,modality)]=data_results
        result_table['{}_{}_data'.format(pipe_name,modality)]=data_results['summary_scores'][0]

        result_store['{}_{}_all'.format(pipe_name,modality)]=all_results
        result_table['{}_{}_all'.format(pipe_name,modality)]=all_results['summary_scores'][0]
result_table.to_csv('drd_results_table_final.csv')

################tests on lockbox for primary analyses##########
#covars only
ML = get_setup_ML(get='all_noqc',test_subjects=test_subs, val_check=3)
all_en_covs_test = test(EN, 'mean_indif', 'covars')

#covars only
ML = get_setup_ML(get='all_noqc',test_subjects=test_subs, val_check=3)
all_en_covs_test5 = test(EN5, 'mean_indif', 'covars')
all_en_covs_test6 = test(EN6, 'mean_indif', 'covars')

#mri vars
ML = get_setup_ML(get='rs',test_subjects=test_subs, val_check=3)
rs_en_data_test = test(EN, 'mean_indif', 'data')
rs_rf_data_test = test(RF, 'mean_indif', 'data')
rs_svm_data_test = test(SVM, 'mean_indif', 'data')

ML = get_setup_ML(get='nback',test_subjects=test_subs, val_check=3)
nb_rf_data_test = test(RF, 'mean_indif', 'data')

ML = get_setup_ML(get='sst',test_subjects=test_subs, val_check=3)
sst_en_data_test = test_shap(EN, 'mean_indif', 'data')
sst_lgb_data_test = test(LGB, 'mean_indif', 'data')

#covars + mri vars
ML = get_setup_ML(get='rs',test_subjects=test_subs, val_check=3)
rs_en_all_test = test(EN, 'mean_indif', 'all')
rs_rf_all_test = test(RF, 'mean_indif', 'all')

#Get beta values for EN whole sample for behavioral vars
covar_beta_weights = all_en_covs_test['FIs'][0]
means = covar_beta_weights.global_df.mean(axis=0)
stds = covar_beta_weights.global_df.std(axis=0)
means.sort_values().to_csv('en_all_covs_betas_1p160.csv')

covar_beta_weights = all_en_covs_test5['FIs'][0]
means = covar_beta_weights.global_df.mean(axis=0)
stds = covar_beta_weights.global_df.std(axis=0)
means.sort_values().to_csv('en_all_covs_betas5.csv')

covar_beta_weights = all_en_covs_test6['FIs'][0]
means = covar_beta_weights.global_df.mean(axis=0)
stds = covar_beta_weights.global_df.std(axis=0)
means.sort_values().to_csv('en_all_covs_betas6.csv')


########################SUPPLEMENTARY ANALYSES###########
#auc analysis
result_store={}
result_table=pd.DataFrame(index=['r2','micro_sd','macro_sd'])
for modality in modalities:
    for pipeline,pipe_name in zip(pipelines,pipe_names):

        ML = get_setup_ML(get=modality,test_subjects=test_subs)

        covar_results = evaluate(pipeline, 'AUCmy', 'covars')
        data_results = evaluate(pipeline, 'AUCmy', 'data')
        all_results = evaluate(pipeline, 'AUCmy', 'all')

        result_store['{}_{}_covs'.format(pipe_name,modality)]=covar_results
        result_table['{}_{}_covs'.format(pipe_name,modality)]=covar_results['summary_scores'][0]

        result_store['{}_{}_data'.format(pipe_name,modality)]=data_results
        result_table['{}_{}_data'.format(pipe_name,modality)]=data_results['summary_scores'][0]

        result_store['{}_{}_all'.format(pipe_name,modality)]=all_results
        result_table['{}_{}_all'.format(pipe_name,modality)]=all_results['summary_scores'][0]
result_table.to_csv('drd_results_table_final_auc.csv')

# different DRD performance exclusions
result_store={}
result_table=pd.DataFrame(index=['r2','micro_sd','macro_sd'])
#val_opts = [4,2,1]
val_opts = [1]
for val_opt in val_opts:
    for modality in modalities:
        for pipeline,pipe_name in zip(pipelines,pipe_names):

            ML = get_setup_ML(get=modality,test_subjects=test_subs,val_check=val_opt)

            covar_results = evaluate(pipeline, 'mean_indif', 'covars')
            data_results = evaluate(pipeline, 'mean_indif', 'data')
            all_results = evaluate(pipeline, 'mean_indif', 'all')

            result_store['{}_{}_covs_val{}'.format(pipe_name,modality,val_opt)]=covar_results
            result_table['{}_{}_covs_val{}'.format(pipe_name,modality,val_opt)]=covar_results['summary_scores'][0]

            result_store['{}_{}_data_val{}'.format(pipe_name,modality,val_opt)]=data_results
            result_table['{}_{}_data_val{}'.format(pipe_name,modality,val_opt)]=data_results['summary_scores'][0]

            result_store['{}_{}_all_val{}'.format(pipe_name,modality,val_opt)]=all_results
            result_table['{}_{}_all_val{}'.format(pipe_name,modality,val_opt)]=all_results['summary_scores'][0]
result_table.to_csv('drd_results_table_final_1valcheck.csv')

#site analysis
for modality in modalities:
    for pipeline,pipe_name in zip(pipelines,pipe_names):

        ML = get_setup_ML(get=modality,test_subjects=test_subs)

        covar_results = evaluate(pipeline, 'mean_indif', 'covars')
        data_results = evaluate(pipeline, 'mean_indif', 'data')
        all_results = evaluate(pipeline, 'mean_indif', 'all')

        result_store['{}_{}_covs'.format(pipe_name,modality)]=covar_results
        result_table['{}_{}_covs'.format(pipe_name,modality)]=covar_results['summary_scores'][0]

        result_store['{}_{}_data'.format(pipe_name,modality)]=data_results
        result_table['{}_{}_data'.format(pipe_name,modality)]=data_results['summary_scores'][0]

        result_store['{}_{}_all'.format(pipe_name,modality)]=all_results
        result_table['{}_{}_all'.format(pipe_name,modality)]=all_results['summary_scores'][0]
result_table.to_csv('drd_results_table_final_site.csv')

## Ridge Regression Recursive Feature Elimination
#define piplelines
ridge_rfe = Feat_Selector('rfe', base_model=Model('ridge', params=1))


EN = Model_Pipeline(imputers=imputers,
                          scalers=robust_scaler,
                          transformers=ohe,
                          feat_selectors=ridge_rfe,
                          model = Model('elastic net regressor', params=1),
                          param_search = alt_search)
RF = Model_Pipeline(imputers=imputers,
                          scalers=robust_scaler,
                          transformers=ohe,
                          feat_selectors=ridge_rfe,
                          model = Model('random forest regressor', params=1),
                          param_search = alt_search)

LGB = Model_Pipeline(imputers=imputers,
                          scalers=robust_scaler,
                          transformers=ohe,
                          feat_selectors=ridge_rfe,
                          model = Model('light gbm regressor', params=1),
                          param_search = alt_search)

SVM = Model_Pipeline(imputers=imputers,
                          scalers=robust_scaler,
                          transformers=ohe,
                          feat_selectors=ridge_rfe,
                          model = Model('svm regressor', params=1),
                          param_search = alt_search)

pipelines_rfe = [EN,RF,LGB,SVM]
pipe_names_rfe = ['EN','RF','LGB','SVM']

modalities = ['smri','dmri','rs','nback','sst','mid']

#run models using rfe (use MRI data only because of computational demands)
for modality in modalities:
    for pipeline,pipe_name in zip(pipelines_rfe,pipe_names_rfe):

        ML = get_setup_ML(get=modality,test_subjects=test_subs)

        #covar_results = evaluate(pipeline, 'mean_indif', 'covars')
        data_results = evaluate(pipeline, 'mean_indif', 'data')
        #all_results = evaluate(pipeline, 'mean_indif', 'all')

        #result_store['{}_{}_covs'.format(pipe_name,modality)]=covar_results
        #result_table['{}_{}_covs'.format(pipe_name,modality)]=covar_results['summary_scores'][0]

        result_store['{}_{}_data'.format(pipe_name,modality)]=data_results
        result_table['{}_{}_data'.format(pipe_name,modality)]=data_results['summary_scores'][0]

        #result_store['{}_{}_all'.format(pipe_name,modality)]=all_results
        #result_table['{}_{}_all'.format(pipe_name,modality)]=all_results['summary_scores'][0]
result_table.to_csv('drd_results_table_final_ridge_RFE.csv')

########################Prediction of IQ#######################################
#IQ ANALYSIS
for modality in modalities:
    for pipeline,pipe_name in zip(pipelines,pipe_names):

        ML = get_setup_ML(get=modality,test_subjects=test_subs)

        covar_results = evaluate(pipeline, 'nihtbx_totalcomp_uncorrected', 'covars')
        data_results = evaluate(pipeline, 'nihtbx_totalcomp_uncorrected', 'data')
        all_results = evaluate(pipeline, 'nihtbx_totalcomp_uncorrected', 'all')

        result_store['{}_{}_covs'.format(pipe_name,modality)]=covar_results
        result_table['{}_{}_covs'.format(pipe_name,modality)]=covar_results['summary_scores'][0]

        result_store['{}_{}_data'.format(pipe_name,modality)]=data_results
        result_table['{}_{}_data'.format(pipe_name,modality)]=data_results['summary_scores'][0]

        result_store['{}_{}_all'.format(pipe_name,modality)]=all_results
        result_table['{}_{}_all'.format(pipe_name,modality)]=all_results['summary_scores'][0]
result_table.to_csv('IQ_results_table_final.csv')

################tests on lockbox for IQ##########
#mri vars
ML = get_setup_ML(get='all',test_subjects=test_subs, val_check=3)
all_en_iq_test = test(EN, 'nihtbx_totalcomp_uncorrected', 'data')
ML = get_setup_ML(get='all',test_subjects=test_subs, val_check=3)
all_en_iq_test = test(EN, 'nihtbx_totalcomp_uncorrected', 'data')


################tests on lockbox for supplemental analyses##############
#DRD QC 4 inconsisticies
ML = get_setup_ML(get='sst',test_subjects=test_subs, val_check=4)
sst_EN_data_test_qc4 = test(EN, 'mean_indif', 'data')

#Site as grouping var
from loading_final_site import get_setup_ML, get_cat_vars, load_data #pull in alternate loading file
ML = get_setup_ML(get='sst',test_subjects=test_subs, val_check=3)
iq_en_mid_test = test(EN, 'mean_indif', 'data')
from loading_final import get_setup_ML, get_cat_vars, load_data #return to normal loading file

#IQ
ML = get_setup_ML(get='smri',test_subjects=test_subs, val_check=3)
iq_en_smri_test = test(EN, 'nihtbx_totalcomp_uncorrected', 'data')

ML = get_setup_ML(get='dmri',test_subjects=test_subs, val_check=3)
iq_en_dmri_test = test(EN, 'nihtbx_totalcomp_uncorrected', 'data')

ML = get_setup_ML(get='rs',test_subjects=test_subs, val_check=3)
iq_en_rs_test = test(SVM, 'nihtbx_totalcomp_uncorrected', 'data')

ML = get_setup_ML(get='nback',test_subjects=test_subs, val_check=3)
iq_en_nb_test = test(EN, 'nihtbx_totalcomp_uncorrected', 'data')

ML = get_setup_ML(get='mid',test_subjects=test_subs, val_check=3)
iq_en_mid_test = test(EN, 'nihtbx_totalcomp_uncorrected', 'data')

#################bivariate regressions for covariates vars############
data_all=load_data(get='all_noqc',val_check=3)
data = data_all[0]
cats = ['devhx_21_speech_dev_p', 'sex','ksads_back_grades_in_school_p',
       'married.bl','household.income.bl']
for cat in cats:
    dummy = pd.get_dummies(data[cat])
    data = pd.concat([data.reset_index(drop=True), dummy.reset_index(drop=True)], axis=1)
result_table=pd.DataFrame(index=['screentime_weekend_p', 'screentime_ss_weekday', 'screentime_week_p',
      'su_caff_ss_sum_calc', 'upps_ss_positive_urgency', 'nihtbx_picvocab_uncorrected',
       'cbcl_scr_syn_rulebreak_r', 'cbcl_scr_syn_withdep_r', 'resiliency_opposite_sex_friends_cat',
       'devhx_milestones',  'resiliency_same_sex_friends_cat', 'nihtbx_cryst_uncorrected', 
    'sports_activity_activities_p_ind_sport', 'parental_monitoring_ss_mean',
      'prosocial_ss_mean', 'nihtbx_cardsort_uncorrected', 'F', 'M', 'About average', 'Much earlier',
       'Much later', 'Somewhat earlier', 'Somewhat later', "A's / Excellent", "B's / Good",
       "C's / Average", "D's / Below Average", "F's / Struggling a lot",
       'ungraded', 'no', 'yes', '[<50K]', '[>=100K]', '[>=50K & <100K]'], columns=['B','SE','p','R2'])
ivs = ['screentime_weekend_p', 'screentime_ss_weekday', 'screentime_week_p',
      'su_caff_ss_sum_calc', 'upps_ss_positive_urgency', 'nihtbx_picvocab_uncorrected',
       'cbcl_scr_syn_rulebreak_r', 'cbcl_scr_syn_withdep_r', 'resiliency_opposite_sex_friends_cat',
       'devhx_milestones',  'resiliency_same_sex_friends_cat', 'nihtbx_cryst_uncorrected', 
    'sports_activity_activities_p_ind_sport', 'parental_monitoring_ss_mean',
      'prosocial_ss_mean', 'nihtbx_cardsort_uncorrected', 'F', 'M', 'About average', 'Much earlier',
       'Much later', 'Somewhat earlier', 'Somewhat later', "A's / Excellent", "B's / Good",
       "C's / Average", "D's / Below Average", "F's / Struggling a lot",
       'ungraded', 'no', 'yes', '[<50K]', '[>=100K]', '[>=50K & <100K]']
i = 0
for iv in ivs:
    X = pd.to_numeric(data[iv])
    Y = data['mean_indif']
    XY = pd.concat([X, Y], axis=1)
    XY = XY.dropna()
    lm = scipy.stats.linregress(XY.iloc[:,0],XY.iloc[:,1])
    result_table.loc[iv, 'B'] = lm[0]
    result_table.loc[iv, 'SE'] = lm[4]
    result_table.loc[iv, 'p'] = lm[3]
    result_table.loc[iv, 'R2'] = lm[2]**2
    i += 1
data = data_all[0]
cats = ['devhx_20_motor_dev_p']
for cat in cats:
    dummy = pd.get_dummies(data[cat])
    data = pd.concat([data.reset_index(drop=True), dummy.reset_index(drop=True)], axis=1)
result_table2=pd.DataFrame(index = ['Much earlier','Much later', 'Somewhat earlier', 'Somewhat later'], 
                                   columns = ['B','SE','p','R2'])
ivs = ['About average','Much earlier','Much later', 'Somewhat earlier', 'Somewhat later']
i = 0
for iv in ivs:
    X = pd.to_numeric(data[iv])
    Y = data['mean_indif']
    XY = pd.concat([X, Y], axis=1)
    XY = XY.dropna()
    lm = scipy.stats.linregress(XY.iloc[:,0],XY.iloc[:,1])
    result_table2.loc[iv, 'B'] = lm[0]
    result_table2.loc[iv, 'SE'] = lm[4]
    result_table2.loc[iv, 'p'] = lm[3]
    result_table2.loc[iv, 'R2'] = lm[2]**2
    i += 1
result_table = result_table.sort_values('p', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last', 
                      ignore_index=False, key=None)
fdr=sm.stats.multitest.fdrcorrection(result_table['p'], alpha = 0.05, method='indep', is_sorted=False)
result_table['fdr'] = fdr[0]
result_table['fdr_p'] = fdr[1]
result_table.to_csv('univariate_stats.csv')

###########################different hp search approaches######################
param_search = Param_Search(search_type='RandomSearch',
                            n_iter= 60)
alt_search1 = Param_Search(search_type='RandomSearch',
                          n_iter= 100)
alt_search2 = Param_Search(search_type='RandomSearch',
                          n_iter= 200)
alt_search3 = Param_Search(search_type='HammersleySearchPlusMiddlePoint',
                          n_iter= 200)
alt_search4 = Param_Search(search_type='TwoPointsDE',
                          n_iter= 200)
alt_search5 = Param_Search(search_type='DiscreteOnePlusOne',
                          n_iter= 100)
alt_search6 = Param_Search(search_type='DiscreteOnePlusOne',
                          n_iter= 200)

EN = Model_Pipeline(imputers=imputers,
                          scalers=robust_scaler,
                          transformers=ohe,
                          model = Model('elastic net regressor', params=1),
                          param_search = param_search)
RF = Model_Pipeline(imputers=imputers,
                          scalers=robust_scaler,
                          transformers=ohe,
                          model = Model('random forest regressor', params=1),
                          param_search = param_search)

LGB = Model_Pipeline(imputers=imputers,
                          scalers=robust_scaler,
                          transformers=ohe,
                          model = Model('light gbm regressor', params=1),
                          param_search = param_search)

SVM = Model_Pipeline(imputers=imputers,
                          scalers=robust_scaler,
                          transformers=ohe,
                          model = Model('svm regressor', params=1),
                          param_search = param_search)

EN1 = Model_Pipeline(imputers=imputers,
                          scalers=robust_scaler,
                          transformers=ohe,
                          model = Model('elastic net regressor', params=1),
                          param_search = alt_search1)
RF1 = Model_Pipeline(imputers=imputers,
                          scalers=robust_scaler,
                          transformers=ohe,
                          model = Model('random forest regressor', params=1),
                          param_search = alt_search1)

LGB1 = Model_Pipeline(imputers=imputers,
                          scalers=robust_scaler,
                          transformers=ohe,
                          model = Model('light gbm regressor', params=1),
                          param_search = alt_search1)

SVM1 = Model_Pipeline(imputers=imputers,
                          scalers=robust_scaler,
                          transformers=ohe,
                          model = Model('svm regressor', params=1),
                          param_search = alt_search1)

EN2 = Model_Pipeline(imputers=imputers,
                          scalers=robust_scaler,
                          transformers=ohe,
                          model = Model('elastic net regressor', params=1),
                          param_search = alt_search2)
RF2 = Model_Pipeline(imputers=imputers,
                          scalers=robust_scaler,
                          transformers=ohe,
                          model = Model('random forest regressor', params=1),
                          param_search = alt_search2)

LGB2 = Model_Pipeline(imputers=imputers,
                          scalers=robust_scaler,
                          transformers=ohe,
                          model = Model('light gbm regressor', params=1),
                          param_search = alt_search2)

SVM2 = Model_Pipeline(imputers=imputers,
                          scalers=robust_scaler,
                          transformers=ohe,
                          model = Model('svm regressor', params=1),
                          param_search = alt_search2)

EN3 = Model_Pipeline(imputers=imputers,
                          scalers=robust_scaler,
                          transformers=ohe,
                          model = Model('elastic net regressor', params=1),
                          param_search = alt_search3)
RF3 = Model_Pipeline(imputers=imputers,
                          scalers=robust_scaler,
                          transformers=ohe,
                          model = Model('random forest regressor', params=1),
                          param_search = alt_search3)

LGB3 = Model_Pipeline(imputers=imputers,
                          scalers=robust_scaler,
                          transformers=ohe,
                          model = Model('light gbm regressor', params=1),
                          param_search = alt_search3)

SVM3 = Model_Pipeline(imputers=imputers,
                          scalers=robust_scaler,
                          transformers=ohe,
                          model = Model('svm regressor', params=1),
                          param_search = alt_search3)

EN4 = Model_Pipeline(imputers=imputers,
                          scalers=robust_scaler,
                          transformers=ohe,
                          model = Model('elastic net regressor', params=1),
                          param_search = alt_search4)
RF4 = Model_Pipeline(imputers=imputers,
                          scalers=robust_scaler,
                          transformers=ohe,
                          model = Model('random forest regressor', params=1),
                          param_search = alt_search4)

LGB4 = Model_Pipeline(imputers=imputers,
                          scalers=robust_scaler,
                          transformers=ohe,
                          model = Model('light gbm regressor', params=1),
                          param_search = alt_search4)

SVM4 = Model_Pipeline(imputers=imputers,
                          scalers=robust_scaler,
                          transformers=ohe,
                          model = Model('svm regressor', params=1),
                          param_search = alt_search4)

EN5 = Model_Pipeline(imputers=imputers,
                          scalers=robust_scaler,
                          transformers=ohe,
                          model = Model('elastic net regressor', params=1),
                          param_search = alt_search5)
RF5 = Model_Pipeline(imputers=imputers,
                          scalers=robust_scaler,
                          transformers=ohe,
                          model = Model('random forest regressor', params=1),
                          param_search = alt_search5)

LGB5 = Model_Pipeline(imputers=imputers,
                          scalers=robust_scaler,
                          transformers=ohe,
                          model = Model('light gbm regressor', params=1),
                          param_search = alt_search5)

SVM5 = Model_Pipeline(imputers=imputers,
                          scalers=robust_scaler,
                          transformers=ohe,
                          model = Model('svm regressor', params=1),
                          param_search = alt_search5)

EN6 = Model_Pipeline(imputers=imputers,
                          scalers=robust_scaler,
                          transformers=ohe,
                          model = Model('elastic net regressor', params=1),
                          param_search = alt_search6)
RF6 = Model_Pipeline(imputers=imputers,
                          scalers=robust_scaler,
                          transformers=ohe,
                          model = Model('random forest regressor', params=1),
                          param_search = alt_search6)

LGB6 = Model_Pipeline(imputers=imputers,
                          scalers=robust_scaler,
                          transformers=ohe,
                          model = Model('light gbm regressor', params=1),
                          param_search = alt_search6)

SVM6 = Model_Pipeline(imputers=imputers,
                          scalers=robust_scaler,
                          transformers=ohe,
                          model = Model('svm regressor', params=1),
                          param_search = alt_search6)

pipelines = [EN,RF,LGB,SVM]
pipelines1 = [EN1,RF1,LGB1,SVM1]
pipelines2 = [EN2,RF2,LGB2,SVM2]
pipelines3 = [EN3,RF3,LGB3,SVM3]
pipelines4 = [EN4,RF4,LGB4,SVM4]
pipelines5 = [EN5,RF5,LGB5,SVM5]
pipelines6 = [EN6,RF6,LGB6,SVM6]
pipe_names = ['EN','RF','LGB','SVM']

modalities = ['smri','dmri','rs','nback','sst','mid']

####################Different Hyperparam Search############################
result_store={}
result_table=pd.DataFrame(index=['r2','micro_sd','macro_sd'])
for modality in modalities:
    for pipeline,pipe_name in zip(pipelines,pipe_names):

        ML = get_setup_ML(get=modality,test_subjects=test_subs)

        covar_results = evaluate(pipeline, 'mean_indif', 'covars')
        data_results = evaluate(pipeline, 'mean_indif', 'data')
        all_results = evaluate(pipeline, 'mean_indif', 'all')

        result_store['{}_{}_covs'.format(pipe_name,modality)]=covar_results
        result_table['{}_{}_covs'.format(pipe_name,modality)]=covar_results['summary_scores'][0]

        result_store['{}_{}_data'.format(pipe_name,modality)]=data_results
        result_table['{}_{}_data'.format(pipe_name,modality)]=data_results['summary_scores'][0]

        result_store['{}_{}_all'.format(pipe_name,modality)]=all_results
        result_table['{}_{}_all'.format(pipe_name,modality)]=all_results['summary_scores'][0]
result_table.to_csv('drd_results_table_hpsearch_def.csv')


result_store1={}
result_table1=pd.DataFrame(index=['r2','micro_sd','macro_sd'])
for modality in modalities:
    for pipeline,pipe_name in zip(pipelines1,pipe_names):

        ML = get_setup_ML(get=modality,test_subjects=test_subs)

        covar_results = evaluate(pipeline, 'mean_indif', 'covars')
        data_results = evaluate(pipeline, 'mean_indif', 'data')
        all_results = evaluate(pipeline, 'mean_indif', 'all')

        result_store1['{}_{}_covs'.format(pipe_name,modality)]=covar_results
        result_table1['{}_{}_covs'.format(pipe_name,modality)]=covar_results['summary_scores'][0]

        result_store1['{}_{}_data'.format(pipe_name,modality)]=data_results
        result_table1['{}_{}_data'.format(pipe_name,modality)]=data_results['summary_scores'][0]

        result_store1['{}_{}_all'.format(pipe_name,modality)]=all_results
        result_table1['{}_{}_all'.format(pipe_name,modality)]=all_results['summary_scores'][0]
result_table1.to_csv('drd_results_table_hpsearch_1.csv')


result_store2={}
result_table2=pd.DataFrame(index=['r2','micro_sd','macro_sd'])
for modality in modalities:
    for pipeline,pipe_name in zip(pipelines2,pipe_names):

        ML = get_setup_ML(get=modality,test_subjects=test_subs)

        covar_results = evaluate(pipeline, 'mean_indif', 'covars')
        data_results = evaluate(pipeline, 'mean_indif', 'data')
        all_results = evaluate(pipeline, 'mean_indif', 'all')

        result_store2['{}_{}_covs'.format(pipe_name,modality)]=covar_results
        result_table2['{}_{}_covs'.format(pipe_name,modality)]=covar_results['summary_scores'][0]

        result_store2['{}_{}_data'.format(pipe_name,modality)]=data_results
        result_table2['{}_{}_data'.format(pipe_name,modality)]=data_results['summary_scores'][0]

        result_store2['{}_{}_all'.format(pipe_name,modality)]=all_results
        result_table2['{}_{}_all'.format(pipe_name,modality)]=all_results['summary_scores'][0]
result_table2.to_csv('drd_results_table_hpsearch_2.csv')


result_store3={}
result_table3=pd.DataFrame(index=['r2','micro_sd','macro_sd'])
for modality in modalities:
    for pipeline,pipe_name in zip(pipelines3,pipe_names):

        ML = get_setup_ML(get=modality,test_subjects=test_subs)

        covar_results = evaluate(pipeline, 'mean_indif', 'covars')
        data_results = evaluate(pipeline, 'mean_indif', 'data')
        all_results = evaluate(pipeline, 'mean_indif', 'all')

        result_store3['{}_{}_covs'.format(pipe_name,modality)]=covar_results
        result_table3['{}_{}_covs'.format(pipe_name,modality)]=covar_results['summary_scores'][0]

        result_store3['{}_{}_data'.format(pipe_name,modality)]=data_results
        result_table3['{}_{}_data'.format(pipe_name,modality)]=data_results['summary_scores'][0]

        result_store3['{}_{}_all'.format(pipe_name,modality)]=all_results
        result_table3['{}_{}_all'.format(pipe_name,modality)]=all_results['summary_scores'][0]
result_table3.to_csv('drd_results_table_hpsearch_3.csv')


result_store4={}
result_table4=pd.DataFrame(index=['r2','micro_sd','macro_sd'])
for modality in modalities:
    for pipeline,pipe_name in zip(pipelines4,pipe_names):

        ML = get_setup_ML(get=modality,test_subjects=test_subs)

        covar_results = evaluate(pipeline, 'mean_indif', 'covars')
        data_results = evaluate(pipeline, 'mean_indif', 'data')
        all_results = evaluate(pipeline, 'mean_indif', 'all')

        result_store4['{}_{}_covs'.format(pipe_name,modality)]=covar_results
        result_table4['{}_{}_covs'.format(pipe_name,modality)]=covar_results['summary_scores'][0]

        result_store4['{}_{}_data'.format(pipe_name,modality)]=data_results
        result_table4['{}_{}_data'.format(pipe_name,modality)]=data_results['summary_scores'][0]

        result_store4['{}_{}_all'.format(pipe_name,modality)]=all_results
        result_table4['{}_{}_all'.format(pipe_name,modality)]=all_results['summary_scores'][0]
result_table4.to_csv('drd_results_table_hpsearch_4.csv')

result_store5={}
result_table5=pd.DataFrame(index=['r2','micro_sd','macro_sd'])
for modality in modalities:
    for pipeline,pipe_name in zip(pipelines5,pipe_names):

        ML = get_setup_ML(get=modality,test_subjects=test_subs)

        covar_results = evaluate(pipeline, 'mean_indif', 'covars')
        data_results = evaluate(pipeline, 'mean_indif', 'data')
        all_results = evaluate(pipeline, 'mean_indif', 'all')

        result_store5['{}_{}_covs'.format(pipe_name,modality)]=covar_results
        result_table5['{}_{}_covs'.format(pipe_name,modality)]=covar_results['summary_scores'][0]

        result_store5['{}_{}_data'.format(pipe_name,modality)]=data_results
        result_table5['{}_{}_data'.format(pipe_name,modality)]=data_results['summary_scores'][0]

        result_store5['{}_{}_all'.format(pipe_name,modality)]=all_results
        result_table5['{}_{}_all'.format(pipe_name,modality)]=all_results['summary_scores'][0]
result_table5.to_csv('drd_results_table_hpsearch_5.csv')

result_store6={}
result_table6=pd.DataFrame(index=['r2','micro_sd','macro_sd'])
for modality in modalities:
    for pipeline,pipe_name in zip(pipelines6,pipe_names):

        ML = get_setup_ML(get=modality,test_subjects=test_subs)

        covar_results = evaluate(pipeline, 'mean_indif', 'covars')
        data_results = evaluate(pipeline, 'mean_indif', 'data')
        all_results = evaluate(pipeline, 'mean_indif', 'all')

        result_store6['{}_{}_covs'.format(pipe_name,modality)]=covar_results
        result_table6['{}_{}_covs'.format(pipe_name,modality)]=covar_results['summary_scores'][0]

        result_store6['{}_{}_data'.format(pipe_name,modality)]=data_results
        result_table6['{}_{}_data'.format(pipe_name,modality)]=data_results['summary_scores'][0]

        result_store6['{}_{}_all'.format(pipe_name,modality)]=all_results
        result_table6['{}_{}_all'.format(pipe_name,modality)]=all_results['summary_scores'][0]
result_table6.to_csv('drd_results_table_hpsearch_6.csv')

################tests on lockbox for alternate searches##########
#covars only
ML = get_setup_ML(get='all_noqc',test_subjects=test_subs, val_check=3)
all_en_covs_test_rs60 = test(EN, 'mean_indif', 'covars')
all_en_covs_test_rs100 = test(EN1, 'mean_indif', 'covars')
all_en_covs_test_rs200 = test(EN2, 'mean_indif', 'covars')

#Get beta values for EN whole sample for behavioral vars to ensure comparable to primary anlaysis
covar_beta_weights = all_en_covs_test_rs60['FIs'][0]
means = covar_beta_weights.global_df.mean(axis=0)
stds = covar_beta_weights.global_df.std(axis=0)
means.sort_values().to_csv('en_all_covs_betas60.csv')

covar_beta_weights = all_en_covs_test_rs100['FIs'][0]
means = covar_beta_weights.global_df.mean(axis=0)
stds = covar_beta_weights.global_df.std(axis=0)
means.sort_values().to_csv('en_all_covs_betas100.csv')

covar_beta_weights = all_en_covs_test_rs200['FIs'][0]
means = covar_beta_weights.global_df.mean(axis=0)
stds = covar_beta_weights.global_df.std(axis=0)
means.sort_values().to_csv('en_all_covs_betas200.csv')
