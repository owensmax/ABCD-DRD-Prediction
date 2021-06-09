import pandas as pd #v 1.1.5
import numpy as np #v 1.18.1
from BPt import * #v 1.3.4

data_file_loc = 'drd_data_dest.csv'
cols_file_loc = 'drd_columns_final.csv'
brain_feats_loc = 'drd_brain_features.txt'
indif_varnames_loc = 'indif_varnames.txt'
philips_loc = 'philips.txt'
sst_exclude_loc = 'sst_exclude.txt'
sst_good_loc = 'sst_good.txt'
nback_good_loc = 'nback_good.txt'
mid_good_loc = 'mid_good.txt'

def get_cat_vars():

    ns = pd.read_csv(cols_file_loc )
    behav_dtypes = ns['vartypes'].tolist()
    cat_vars = [ns.loc[i, 'varnames'] for i in range(len(behav_dtypes)) if behav_dtypes[i] == 'c']

    return cat_vars

#set up functions
def summer(var_ls,data):
    output_name=data.loc[:, var_ls].sum(axis=1)
    return output_name
def sum_clip(var_ls,data):
    output_name=data.loc[:, var_ls].sum(axis=1)
    output_name=output_name.clip(upper=1)
    return output_name
def dropper(vars_ls,data):
    for v in vars_ls:
        data=data.drop([v],axis=1) 
    return data
def replacer(vars_ls,rep_dic,data):
    for var in vars_ls:
        for val in rep_dic.keys():
            data[var] = data[var].replace(val, rep_dic[val])
    return data
def na_filler(var_ls,fillnum,data):
    for var in var_ls:
        data[var]=data[var].fillna(fillnum)
    return data

def load_data(get='mri_nback',val_check=3):

    file=pd.read_csv(data_file_loc)

    data=file
    ns=pd.read_csv(cols_file_loc)
    cash_choice_alone=ns[ns['varnames']=='neurocog_cash_choice_task']
    ns=ns[ns['varnames']!='neurocog_cash_choice_task']
    ns=ns[ns['varnames']!='nihtbx_totalcomp_uncorrected']
    behav_features=ns['varnames'].tolist()
    behav_dtypes=ns['vartypes'].tolist()
    #with open('apriori_brain_features.txt') as f:
    with open(brain_feats_loc) as f:
        brain_features = f.read().splitlines()
    with open(indif_varnames_loc) as f:
        targets = f.read().splitlines()
    proc_vars=['src_subject_id','eventname','fsqc_qc','iqc_dmri_good_ser',
            'rsfmri_var_ntpoints',"iqc_rsfmri_good_ser","rel_family_id","mri_info_device.serial.number"]
    mri_qc_features=['iqc_rsfmri_all_mean_motion','iqc_dmri_all_mean_motion','tfmri_mid_all_beta_mean.motion',
    'tfmri_nback_all_beta_mean.motion','tfmri_sst_all_beta_mean.motion']
    allvars=proc_vars+behav_features+brain_features+mri_qc_features+targets
    sid=['src_subject_id']
    target='AUCmy'


    activities_vars=['sports_activity_activities_p___0','sports_activity_activities_p___1',
                 'sports_activity_activities_p___2','sports_activity_activities_p___3',
                 'sports_activity_activities_p___4','sports_activity_activities_p___5',
                 'sports_activity_activities_p___6','sports_activity_activities_p___7',
                 'sports_activity_activities_p___8','sports_activity_activities_p___9',
                 'sports_activity_activities_p___10','sports_activity_activities_p___11',
                 'sports_activity_activities_p___12','sports_activity_activities_p___13',
                 'sports_activity_activities_p___14','sports_activity_activities_p___15',
                 'sports_activity_activities_p___16','sports_activity_activities_p___17',
                 'sports_activity_activities_p___18','sports_activity_activities_p___19',
                 'sports_activity_activities_p___20','sports_activity_activities_p___21',
                 'sports_activity_activities_p___22','sports_activity_activities_p___23',
                 'sports_activity_activities_p___24','sports_activity_activities_p___25',
                 'sports_activity_activities_p___26','sports_activity_activities_p___27',
                 'sports_activity_activities_p___28']

    team_sport_vars=['sports_activity_activities_p___1','sports_activity_activities_p___2',
                'sports_activity_activities_p___4','sports_activity_activities_p___5',
                'sports_activity_activities_p___7','sports_activity_activities_p___11',
                'sports_activity_activities_p___12','sports_activity_activities_p___15',
                'sports_activity_activities_p___21']
    ind_sport_vars=['sports_activity_activities_p___3',
            'sports_activity_activities_p___6','sports_activity_activities_p___8',
            'sports_activity_activities_p___9','sports_activity_activities_p___10',
            'sports_activity_activities_p___13','sports_activity_activities_p___14',
            'sports_activity_activities_p___16','sports_activity_activities_p___17',
                'sports_activity_activities_p___18','sports_activity_activities_p___19',
                'sports_activity_activities_p___20','sports_activity_activities_p___22',
            'sports_activity_activities_p___27']
    performance_vars=['sports_activity_activities_p___0','sports_activity_activities_p___23',
                    'sports_activity_activities_p___24','sports_activity_activities_p___25']
    hobbies_vars=['sports_activity_activities_p___26','sports_activity_activities_p___28']

    #set no to 0 and yes 1 (instead of no=1 and yes=2)
    for v in activities_vars:
        data[v]=data[v]-1
        
    data['sports_activity_activities_p_team_sport']=summer(team_sport_vars,data)
    data['sports_activity_activities_p_ind_sport']=summer(ind_sport_vars,data)
    data['sports_activity_activities_p_performance']=summer(performance_vars,data)
    data['sports_activity_activities_p_hobbies']=summer(hobbies_vars,data)

    data=dropper(activities_vars,data)

    #fill nas for tlfb
    data=na_filler(['su_tlfb_cal_scr_num_events'],0,data)

    #make aggregate puberty scale
    pub_vars=['pubertdev_ss_female_category','pubertdev_ss_male_category']
    data=na_filler(pub_vars,0,data)
    data['puberty']=summer(pub_vars,data)
    data=dropper(pub_vars,data)

    #medhx converted in R: 1  2  4  3 NA  5  6 = 0 1 time 3-4 times  2 times    <NA>  5-9 times  >10+ times
    #make ER composite
    medhx_vars=['medhx_ss_4b_times_er_past_yr_p','medhx_ss_5b_times_er_before_past_yr_p']
    data['medhx_er_composite']=summer(medhx_vars,data)
    data=dropper(medhx_vars,data)

    #turn all na, refuse, and don't know into nos (no=1)
    #make family history composite
    famhx_vars=['famhx_7_yes_no_p','famhx_8_yes_no_p','famhx_9_yes_no_p','famhx_10_yes_no_p',
                'famhx_11_yes_no_p','famhx_12_yes_no_p','famhx_13_yes_no_p']
    rep_dic={3:1,4:1}
    data=replacer(famhx_vars,rep_dic,data)
    data['famhx_total']=summer(famhx_vars,data)
    data=dropper(famhx_vars,data)

    #sum of mother problems during birth vars
    devhx_vars=['devhx_10a_severe_nausea_p','devhx_10b_heavy_bleeding_p',
                'devhx_10c_eclampsia_p','devhx_10e_persist_proteinuria_p','devhx_10d_gall_bladder_p',
            'devhx_10f_rubella_p','devhx_10g_severe_anemia_p','devhx_10h_urinary_infections_p',
                'devhx_10i_diabetes_p','devhx_10j_high_blood_press_p','devhx_10k_problems_placenta_p',
            'devhx_10l_accident_injury_p','devhx_10m_other_p']
    rep_dic={"No":0,"Don't know":0,"Yes":1}
    data=replacer(devhx_vars,rep_dic,data)
    data['devhx_mother_probs']=summer(devhx_vars,data)
    data=dropper(devhx_vars,data)
        
    #sum of distress of birth vars
    devhx_vars=['devhx_14a_blue_birth_p','devhx_14b_slow_heart_beat_p','devhx_14c_did_not_breathe_p',
    'devhx_14d_convulsions_p','devhx_14e_jaundice_p','devhx_14f_oxygen_p',
    'devhx_14g_blood_transfuse_p','devhx_14h_rh_incompatible_p']
    rep_dic={"No":0,"Don't know":0,"Yes":1}
    data=replacer(devhx_vars,rep_dic,data)
    data['devhx_distress_at_birth']=summer(devhx_vars,data)
    data=dropper(devhx_vars,data)
    
    #aggregation of milestone meeting
    #have added up months at which each occured as a continuous measure of development speed
    #roll over late at 6 months, sit CDC Average = 9 months, walk CDC Average = 18 months,
    #firstword CDC average = 12
    devhx_vars=['devhx_19a_mnths_roll_over_p','devhx_19b_mnths_sit_p',
                'devhx_19c_mnths_walk_p','devhx_19d_first_word_p']
    rep_dic={"No":0,"Don't know":0,"Yes":1}
    data=replacer(devhx_vars,rep_dic,data)
    data['devhx_milestones']=summer(devhx_vars,data)
    data=dropper(devhx_vars,data)

    data=dropper(['devhx_ss_8_her_morph_amt_p'],data)     
        
    #sum up number of friends
    resiliency_vars=['resiliency_5a','resiliency_5b','resiliency_6a','resiliency_6b']
    data['num_friends']=summer(resiliency_vars,data)
    data['num_friends']=data['resiliency_5a']+data['resiliency_5b']+data['resiliency_6a']+data['resiliency_6b']
    data['male_friends']=summer(resiliency_vars[0:1],data)
    data['female_friends']=summer(resiliency_vars[2:3],data)
    data=dropper(resiliency_vars,data)
    data['same_sex_friends'] = np.where(data['sex']=='M', data['male_friends'], data['female_friends'])
    data['opposite_sex_friends'] = np.where(data['sex']=='M', data['female_friends'], data['male_friends'])
    data=dropper(['male_friends','female_friends'],data)
    data['resiliency_num_friends_cat']=pd.qcut(data['num_friends'],q=5,labels=[1,2,3,4,5])
    data['resiliency_same_sex_friends_cat']=pd.qcut(data['same_sex_friends'],q=5,labels=[1,2,3,4,5])
    data['resiliency_opposite_sex_friends_cat']=pd.qcut(data['opposite_sex_friends'],q=5,labels=[1,2,3,4,5])
    data=dropper(['num_friends','same_sex_friends','opposite_sex_friends'],data)

    rep_dic={0:'No'}
    data=replacer(['ksads_back_c_best_friend_p'],rep_dic,data)

    #make trauma composite
    tra_vars=['ksads_ptsd_raw_754_p','ksads_ptsd_raw_755_p','ksads_ptsd_raw_756_p','ksads_ptsd_raw_757_p',
    'ksads_ptsd_raw_758_p','ksads_ptsd_raw_759_p','ksads_ptsd_raw_760_p','ksads_ptsd_raw_761_p',
    'ksads_ptsd_raw_762_p','ksads_ptsd_raw_763_p','ksads_ptsd_raw_764_p','ksads_ptsd_raw_765_p',
    'ksads_ptsd_raw_766_p','ksads_ptsd_raw_767_p','ksads_ptsd_raw_768_p','ksads_ptsd_raw_769_p',
    'ksads_ptsd_raw_770_p']
    rep_dic={"No":1,"Don't know":1,"Yes":2}

    data=replacer(tra_vars,rep_dic,data)
    for i in range(0,len(tra_vars)):
        data[tra_vars[i]]=data[tra_vars[i]]-1
    data['ksads_ptsd_composite']=sum_clip(tra_vars,data)
    data=dropper(tra_vars,data)

    #Make ksads back items numerical
    tra_vars=['ksads_back_c_bully_p','ksads_back_c_mh_sa_p','ksads_back_c_best_friend_p',
            'ksads_back_c_drop_in_grades_p']
    rep_dic={"No":0,"Don't know":0,"Not sure":0,"Yes":1}
    data=replacer(tra_vars,rep_dic,data)

    #make broader ksads scales
    ksads_depressive_composite=(['ksads_1_840_p','ksads_1_841_p','ksads_1_842_p',
    'ksads_1_843_p','ksads_1_845_p','ksads_1_846_p','ksads_1_847_p','ksads_1_840_t','ksads_1_841_t',
    'ksads_1_842_t','ksads_1_843_t','ksads_1_844_t','ksads_1_845_t','ksads_1_846_t','ksads_1_847_t'])
    data['ksads_depressive_composite']=sum_clip(ksads_depressive_composite,data)
    data=dropper(ksads_depressive_composite,data)
    data['ksads_depressive_composite']=data['ksads_depressive_composite'].astype(float)

    ksads_GAD_composite=['ksads_10_869_p','ksads_10_869_t','ksads_10_870_p','ksads_10_870_t','ksads_10_913_p',
    'ksads_10_913_t','ksads_10_914_p','ksads_10_914_t']
    data['ksads_GAD_composite']=sum_clip(ksads_GAD_composite,data)
    data=dropper(ksads_GAD_composite,data)
    data['ksads_GAD_composite']=data['ksads_GAD_composite'].astype(float)
                        
    ksads_OCD_composite=['ksads_11_917_p','ksads_11_918_p','ksads_11_919_p','ksads_11_920_p']
    data['ksads_OCD_composite']=sum_clip(ksads_OCD_composite,data)
    data=dropper(ksads_OCD_composite,data)
    data['ksads_OCD_composite']=data['ksads_OCD_composite'].astype(float)
                        
    ksads_eating_disorder_composite=['ksads_13_929_p','ksads_13_930_p','ksads_13_931_p',
    'ksads_13_932_p','ksads_13_933_p','ksads_13_934_p','ksads_13_935_p','ksads_13_936_p',
    'ksads_13_937_p','ksads_13_938_p','ksads_13_939_p','ksads_13_940_p','ksads_13_941_p',
    'ksads_13_942_p','ksads_13_943_p','ksads_13_944_p']
    data['ksads_eating_disorder_composite']=sum_clip(ksads_eating_disorder_composite,data)
    data=dropper(ksads_eating_disorder_composite,data)
    data['ksads_eating_disorder_composite']=data['ksads_eating_disorder_composite'].astype(float)
                        
    ksads_adhd_composite=['ksads_14_853_p','ksads_14_854_p','ksads_14_855_p','ksads_14_856_p']
    data['ksads_adhd_composite']=sum_clip(ksads_adhd_composite,data)
    data=dropper(ksads_adhd_composite,data)
    data['ksads_adhd_composite']=data['ksads_adhd_composite'].astype(float)
                        
    ksads_cd_composite=['ksads_16_897_p','ksads_16_898_p','ksads_16_899_p','ksads_16_900_p']
    data['ksads_cd_composite']=sum_clip(ksads_cd_composite,data)
    data=dropper(ksads_cd_composite,data)

    ksads_bipolar_composite=['ksads_2_830_p','ksads_2_830_t','ksads_2_831_p','ksads_2_831_t','ksads_2_832_p',
    'ksads_2_832_t','ksads_2_833_p','ksads_2_833_t','ksads_2_834_p','ksads_2_834_t','ksads_2_835_p',
    'ksads_2_835_t','ksads_2_836_p','ksads_2_836_t','ksads_2_837_p','ksads_2_837_t','ksads_2_838_p',
    'ksads_2_838_t','ksads_2_839_p','ksads_2_839_t']
    data['ksads_bipolar_composite']=sum_clip(ksads_bipolar_composite,data)
    data=dropper(ksads_bipolar_composite,data)
    data['ksads_bipolar_composite']=data['ksads_bipolar_composite'].astype(float)
                        
    ksads_sud_composite=('ksads_20_888_p','ksads_20_889_p','ksads_20_890_p','ksads_20_893_p','ksads_20_894_p')
    data['ksads_sud_composite']=sum_clip(ksads_sud_composite,data)
    data=dropper(ksads_sud_composite,data)
    data['ksads_sud_composite']=data['ksads_sud_composite'].astype(float)
                        
    ksads_nssi_composite=['ksads_23_945_p','ksads_23_945_t','ksads_23_956_p','ksads_23_956_t']
    data['ksads_nssi_composite']=sum_clip(ksads_nssi_composite,data)
    data=dropper(ksads_nssi_composite,data)
    data['ksads_nssi_composite']=data['ksads_nssi_composite'].astype(float)
                                
    ksads_suicide_composite=['ksads_23_946_p','ksads_23_946_t','ksads_23_947_p','ksads_23_947_t',
    'ksads_23_948_p','ksads_23_948_t','ksads_23_949_p','ksads_23_950_t','ksads_23_950_p',
    'ksads_23_951_p','ksads_23_951_t','ksads_23_952_p','ksads_23_952_t','ksads_23_953_p','ksads_23_953_t',
    'ksads_23_954_p','ksads_23_954_t','ksads_23_957_p','ksads_23_957_t','ksads_23_958_p','ksads_23_958_t',
    'ksads_23_959_p','ksads_23_959_t','ksads_23_960_p','ksads_23_960_t','ksads_23_961_p','ksads_23_961_t',
    'ksads_23_962_p','ksads_23_962_t','ksads_23_963_p','ksads_23_963_t','ksads_23_964_p','ksads_23_964_t',
    'ksads_23_965_p','ksads_23_965_t']
    data['ksads_suicide_composite']=sum_clip(ksads_suicide_composite,data)
    data=dropper(ksads_suicide_composite,data)
    data['ksads_suicide_composite']=data['ksads_suicide_composite'].astype(float)

    ksads_psychosis_composite=['ksads_4_826_p','ksads_4_827_p','ksads_4_828_p','ksads_4_829_p','ksads_4_849_p',
    'ksads_4_850_p','ksads_4_851_p','ksads_4_852_p']
    data['ksads_psychosis_composite']=sum_clip(ksads_psychosis_composite,data)
    data=dropper(ksads_psychosis_composite,data)
    data['ksads_psychosis_composite']=data['ksads_psychosis_composite'].astype(float)
                        
    ksads_SAD_composite=['ksads_8_863_p','ksads_8_863_t','ksads_8_864_p','ksads_8_864_t','ksads_8_911_p',
    'ksads_8_911_t','ksads_8_912_p','ksads_8_912_t']
    data['ksads_SAD_composite']=sum_clip(ksads_SAD_composite,data)
    data=dropper(ksads_SAD_composite,data)
    data['ksads_SAD_composite']=data['ksads_SAD_composite'].astype(float)

    #acult_vars=['accult_phenx_q4_p','accult_phenx_q5_p']
    #data=na_filler(acult_vars,1,data)
    #data['accult_phenx_q45_p']=summer(acult_vars,data)
    #data['accult_phenx_q45_p']=data['accult_phenx_q45_p']/2
    #data=dropper(acult_vars,data)

    st_vars=['screentime_1_hours_p','screentime_1_minutes_p','screentime_2_hours_p','screentime_2_minutes_p']
    data['screentime_week_p']=(data['screentime_1_hours_p']+(data['screentime_1_minutes_p']/60))
    data['screentime_weekend_p']=(data['screentime_2_hours_p']+(data['screentime_2_minutes_p']/60))
    data=dropper(st_vars,data)

    su_crpf_avail_vars=['su_crpf_avail_1_p','su_crpf_avail_2_p','su_crpf_avail_3_p','su_crpf_avail_4_p',
    'su_crpf_avail_5_p','su_crpf_avail_6_p']
    data['su_crpf_avail_sum']=summer(su_crpf_avail_vars,data)
    data=dropper(su_crpf_avail_vars,data)

    data['devhx_ss_alcohol_avg_p']=summer(['devhx_ss_8_alcohol_avg_p','devhx_ss_9_alcohol_avg_p'],data)
    data['devhx_ss_alcohol_effects_p']=summer(['devhx_ss_8_alcohol_effects_p','devhx_ss_9_alcohol_effects_p'],data)
    data['devhx_ss_alcohol_max_p']=summer(['devhx_ss_8_alcohol_max_p','devhx_ss_9_alcohol_max_p'],data)
    data['devhx_ss_cigs_per_day_p']=summer(['devhx_ss_8_cigs_per_day_p','devhx_ss_9_cigs_per_day_p'],data)
    data['devhx_ss_coc_crack_amt_p']=summer(['devhx_ss_8_coc_crack_amt_p','devhx_ss_9_coc_crack_amt_p'],data)
    data['devhx_ss_marijuana_amt_p']=summer(['devhx_ss_8_marijuana_amt_p','devhx_ss_9_marijuana_amt_p'],data)
    data['devhx_ss_oxycont_amt_p']=summer(['devhx_ss_8_oxycont_amt_p','devhx_ss_9_oxycont_amt_p'],data)

    devhx_vars=['devhx_ss_8_oxycont_amt_p','devhx_ss_8_marijuana_amt_p','devhx_ss_8_alcohol_avg_p',
    'devhx_ss_8_alcohol_effects_p','devhx_ss_8_alcohol_max_p','devhx_ss_8_coc_crack_amt_p',
    'devhx_ss_8_cigs_per_day_p','devhx_ss_9_oxycont_amt_p','devhx_ss_9_marijuana_amt_p','devhx_ss_9_alcohol_avg_p',
    'devhx_ss_9_alcohol_effects_p','devhx_ss_9_alcohol_max_p','devhx_ss_9_coc_crack_amt_p',
    'devhx_ss_9_cigs_per_day_p',]
    data=dropper(devhx_vars,data)

    data['brain_injury_ss_agefirst_p']=data['brain_injury_ss_agefirst_p'].fillna(-1)
    data['brain_injury_ss_agefirst_p'] = data['brain_injury_ss_agefirst_p'].clip(upper=0)
    data['brain_injury_ss_agefirst_p'] = data['brain_injury_ss_agefirst_p'].replace(0,1)
    data['brain_injury_ss_agefirst_p'] = data['brain_injury_ss_agefirst_p'].replace(-1,0)

    data['ksads_back_trans_prob']=data['ksads_back_trans_prob'].fillna('Not at all')
    data['ksads_back_sex_orient_probs']=data['ksads_back_sex_orient_probs'].fillna('Not at all')
    data['ksads_back_c_trans_prob_p']=data['ksads_back_c_trans_prob_p'].fillna('Not at all')
    data['ksads_back_c_gay_prob_p']=data['ksads_back_c_gay_prob_p'].fillna('Not at all')

    #data['via_accult_ss_amer_p']=data['via_accult_ss_amer_p'].fillna(8)
    #data['via_accult_ss_hc_p']=data['via_accult_ss_hc_p'].fillna(0)

    #1 - AUC average POI
    col=data.loc[:,'ddis_scr_val_indif_point_6h':'ddis_scr_val_indif_pnt_5yr']
    data['mean_indif']=col.mean(axis=1)

    #3: AUC - Myerson 2001 method.
    #ddis_scr_val_indif_point_6h,ddis_scr_val_indif_pnt_1da,ddis_scr_val_indif_pnt_1week,
    #ddis_scr_val_indif_pnt_1mth,ddis_scr_val_indif_pnt_3mth,ddis_scr_val_indif_pnt_1yr,
    #ddis_scr_val_indif_pnt_5yr

    #1-.25
    #.75/365
    #6/365
    #23/365
    #60/365
    #365-90
    #275/365
    #365*5
    #1825-365
    #1460/365

    # Interval 1 1-.25=.75; normalized: .75/365= 0.002054794520547945
    # Interval 2 =7-1=6; normalized: 6/365= 0.01643835616438356
    # Interval 3 =30-7=23; normalized: 60/365= 0.06301369863013699   
    # Interval 4 =90-30=60; normalized: 90/365= 0.1643835616438356
    # Interval 5 =365-90=275; normalized: 275/365= 0.7534246575342466   
    # Interval 6 =1825-365=1460; normalized: 1460/365= 4.0 

    data['Int1'] = data['ddis_scr_val_indif_point_6h'] + data['ddis_scr_val_indif_pnt_1da']*0.002054794520547945 
    data['Int2'] = data['ddis_scr_val_indif_pnt_1week']+ data['ddis_scr_val_indif_pnt_1week']*0.01643835616438356
    data['Int3'] = data['ddis_scr_val_indif_pnt_1week'] + data['ddis_scr_val_indif_pnt_1mth']*0.06301369863013699 
    data['Int4'] = data['ddis_scr_val_indif_pnt_1mth'] + data['ddis_scr_val_indif_pnt_3mth']*0.1643835616438356
    data['Int5'] = data['ddis_scr_val_indif_pnt_3mth'] + data['ddis_scr_val_indif_pnt_1yr']*0.7534246575342466    
    data['Int6'] = data['ddis_scr_val_indif_pnt_1yr'] + data['ddis_scr_val_indif_pnt_5yr']*4.0 
    col2=data.loc[:,'Int1':'Int6']

    data['AUCmy']=col2.sum(axis=1)

    del data['Int1']
    del data['Int2']
    del data['Int3']
    del data['Int4']
    del data['Int5']
    del data['Int6']

    auccor=data['mean_indif'].corr(data['AUCmy'],method='pearson')
    print('Correlation of mean POI & Myerson AUC =',auccor)

    data['drd_vc1'] = data['ddis_scr_val_indif_point_6h']-data['ddis_scr_val_indif_pnt_1da']
    data['drd_vc2'] = data['ddis_scr_val_indif_pnt_1da']-data['ddis_scr_val_indif_pnt_1week']
    data['drd_vc3'] = data['ddis_scr_val_indif_pnt_1week']-data['ddis_scr_val_indif_pnt_1mth']
    data['drd_vc4'] = data['ddis_scr_val_indif_pnt_1mth']-data['ddis_scr_val_indif_pnt_3mth']
    data['drd_vc5'] = data['ddis_scr_val_indif_pnt_3mth']-data['ddis_scr_val_indif_pnt_1yr'] 
    data['drd_vc6'] = data['ddis_scr_val_indif_pnt_1yr']-data['ddis_scr_val_indif_pnt_5yr']

    data['drd_val_check'] = 0
    data.loc[data['drd_vc1'] <0, 'drd_val_check'] += 1
    data.loc[data['drd_vc2'] <0, 'drd_val_check'] += 1
    data.loc[data['drd_vc3'] <0, 'drd_val_check'] += 1
    data.loc[data['drd_vc4'] <0, 'drd_val_check'] += 1
    data.loc[data['drd_vc5'] <0, 'drd_val_check'] += 1
    data.loc[data['drd_vc6'] <0, 'drd_val_check'] += 1
    sum(data['drd_val_check'])

    data['drd_val_check_strict'] = 0
    data.loc[data['drd_vc1'] <=0, 'drd_val_check_strict'] += 1
    data.loc[data['drd_vc2'] <=0, 'drd_val_check_strict'] += 1
    data.loc[data['drd_vc3'] <=0, 'drd_val_check_strict'] += 1
    data.loc[data['drd_vc4'] <=0, 'drd_val_check_strict'] += 1
    data.loc[data['drd_vc5'] <=0, 'drd_val_check_strict'] += 1
    data.loc[data['drd_vc6'] <=0, 'drd_val_check_strict'] += 1

    del data['drd_vc1']
    del data['drd_vc2']
    del data['drd_vc3']
    del data['drd_vc4']
    del data['drd_vc5']
    del data['drd_vc6']

    bad_auc1=sum(data['drd_val_check']>0)
    bad_auc2=sum(data['drd_val_check_strict']>0)
    bad_auc3=sum(data['drd_val_check']>1)
    bad_auc4=sum(data['drd_val_check_strict']>1)
    bad_auc5=sum(data['drd_val_check']>2)
    bad_auc6=sum(data['drd_val_check_strict']>2)
    bad_auc7=sum(data['drd_val_check']>3)
    bad_auc8=sum(data['drd_val_check_strict']>3)

    print("POI reversals > 0:",bad_auc1)
    print("POI reversals or flat > 0:",bad_auc2)
    print("POI reversals > 1:",bad_auc3)
    print("POI reversals or flat > 1:",bad_auc4)
    print("POI reversals > 2:",bad_auc5)
    print("POI reversals or flat > 2:",bad_auc6)
    print("POI reversals > 3:",bad_auc7)
    print("POI reversals or flat > 3:",bad_auc8)

    data=data[data['drd_val_check']<val_check]

    data.dropna(subset=["mean_indif"],axis=0,inplace=True)
    data.dropna(subset=["ddis_scr_val_indif_point_6h"],axis=0,inplace=True)
    targets=['mean_indif','AUCmy','nihtbx_totalcomp_uncorrected']

    #drop construct contamination vars and excess missing vars
    contam_vars=['asr_scr_avoidant_r','asr_scr_adhd_r','asr_scr_antisocial_r','asr_scr_depress_r','asr_scr_hyperactive_r','asr_scr_inattention_r',
    'asr_scr_anxdisord_r','asr_scr_somaticpr_r','cbcl_scr_dsm5_adhd_r','cbcl_scr_dsm5_anxdisord_r','cbcl_scr_dsm5_conduct_r',
    'cbcl_scr_dsm5_depress_r','cbcl_scr_dsm5_opposit_r','cbcl_scr_dsm5_somaticpr_r']
    data=dropper(contam_vars,data)

    excess_missing_vars=['bpmt_scr_attention_r', 'bpmt_scr_external_r', 'bpmt_scr_internal_r',
        'devhx_ss_9_her_morph_amt_p', 'ksads_back_c_reg_friend_group_opin_p',
        'ksads_back_c_reg_friend_group_p','devhx_23b_age_wet_bed_p','demo_prtnr_empl_time_p',
            'demo_prnt_empl_time_p','demo_prtnr_empl_p','devhx_10_p']
    data=dropper(excess_missing_vars,data)

    #below are drops added because they lacked sufficient # cases in 2nd category
    new_drops=['demo_gender_id_p','demo_prnt_gender_id_p','demo_prnt_ethn_p','demo_prnt_empl_p','devhx_5_twin_p',
            'devhx_15_days_incubator_p','devhx_13_ceasarian_p','devhx_16_days_high_fever_p',
            'devhx_17_infections_serious_ill_p','ksads_23_949_t','ksads_18_903_p','ksads_1_844_p',
            'devhx_2b_birth_wt_oz_p','ksads_back_c_gay_prob_p','ksads_back_c_trans_p','ksads_back_c_trans_prob_p',
            'ksads_back_sex_orient','ksads_back_sex_orient_probs','ksads_back_trans_id','ksads_back_trans_prob',
            'su_tlfb_cal_scr_num_events','brain_injury_ss_agefirst_p','prq_q1_p','ksads_back_c_det_susp_p',
            'ksads_back_c_school_setting_p','ksads_sud_composite','demo_relig_p']
    data=dropper(new_drops,data)

    #edits coming as late data cleaning at sage advice
    #make smallest category have a decent # in it
    rep_dic={'Yes - less than once a day but more than once a week':'Yes - less than once a day','Yes - less than once a week':'Yes - less than once a day'}
    data=replacer(['devhx_caffeine_11_p'],rep_dic,data)

    rep_dic={'A lot of conflict':'Some/a lot of conflict','Some conflict':'Some/a lot of conflict'}
    data=replacer(['ksads_back_conflict_p'],rep_dic,data)

    rep_dic={'Failing':'Below Average'}
    data=replacer(['ksads_back_c_how_well_school_p'],rep_dic,data)

    rep_dic={'0':0}
    data=replacer(['ksads_back_c_best_friend_p','ksads_back_c_bully_p','ksads_back_c_drop_in_grades_p',
                'ksads_back_c_mh_sa_p'],rep_dic,data)

    data=data.replace("Don't know",np.NaN)
    data=data.replace("Don't Know",np.NaN)
    data=data.replace("Not sure",np.NaN)
    data=data.replace("-1",np.NaN)
    data=data.replace("Decline to answer",np.NaN)

    mri_qc_features=['iqc_rsfmri_all_mean_motion','iqc_dmri_all_mean_motion','tfmri_mid_all_beta_mean.motion',
    'tfmri_nback_all_beta_mean.motion','tfmri_sst_all_beta_mean.motion']
    nback="tfmri_nback"
    smri='smri'
    sst='tfmri_sst'
    rs='rsfmri'
    mid='tfmri_mid'
    dmri='dmri'
    nbm=["tfmri_nback_all_beta_mean.motion"]
    nback_brain_features = [i for i in brain_features if nback in i] 
    smri_brain_features = [i for i in brain_features if smri in i]
    sst_brain_features = [i for i in brain_features if sst in i]
    rs_brain_features = [i for i in brain_features if rs in i]
    mid_brain_features = [i for i in brain_features if mid in i]
    dmri_brain_features = [i for i in brain_features if dmri in i]
    #nback_brain_features=(nback_brain_features+nbm)
    #rs_brain_features=rs_brain_features+['iqc_rsfmri_all_mean_motion']

    #for corrected mri data, use mri instead of data below
    mri=data

    mri=mri[mri['fsqc_qc']=='accept']
    
    mri_smri=mri
    mri_dmri=mri

    with open(philips_loc) as f:
        philips = f.read().splitlines()
    philips
    mri= mri[~mri['src_subject_id'].isin(philips)]
    mri_nback= mri[~mri['src_subject_id'].isin(philips)]
    mri_sst= mri[~mri['src_subject_id'].isin(philips)]
    mri_rs= mri[~mri['src_subject_id'].isin(philips)]
    mri_mid= mri[~mri['src_subject_id'].isin(philips)]

    with open(sst_exclude_loc) as f:
        sst_exclude = f.read().splitlines()
    philips
    mri= mri[~mri['src_subject_id'].isin(sst_exclude)]
    mri_sst= mri_sst[~mri_sst['src_subject_id'].isin(sst_exclude)]

    with open(sst_good_loc) as f:
        sst_good = f.read().splitlines()
    with open(nback_good_loc) as f:
        nback_good = f.read().splitlines()
    with open(mid_good_loc) as f:
        mid_good = f.read().splitlines()
        
    mri_sst= mri_sst[mri_sst['src_subject_id'].isin(sst_good)]
    mri_nback= mri_nback[mri_nback['src_subject_id'].isin(nback_good)]
    mri_mid= mri_mid[mri_mid['src_subject_id'].isin(mid_good)]
    mri= mri[mri['src_subject_id'].isin(sst_good)]
    mri= mri[mri['src_subject_id'].isin(nback_good)]
    mri= mri[mri['src_subject_id'].isin(mid_good)]

    #mri.dropna(subset=["rsfmri_cor_network.gordon_auditory_network.gordon_auditory"],axis=0,inplace=True)
    mri.dropna(subset=["iqc_rsfmri_all_mean_motion"],axis=0,inplace=True)
    mri_rs.dropna(subset=["iqc_rsfmri_all_mean_motion"],axis=0,inplace=True)
    mri_rs=mri_rs[mri_rs['iqc_rsfmri_good_ser']>=1]
    mri=mri[mri['iqc_rsfmri_good_ser']>=1]
    mri_rs=mri_rs[mri_rs['rsfmri_var_ntpoints']>=375]
    mri=mri[mri['rsfmri_var_ntpoints']>=375]

    mri.dropna(subset=["dmri_dti.full.md_fiber.at_ifsfc.rh"],axis=0,inplace=True)
    mri_dmri.dropna(subset=["dmri_dti.full.md_fiber.at_ifsfc.rh"],axis=0,inplace=True)
    mri_dmri=mri_dmri[mri_dmri['iqc_dmri_good_ser']>=1]
    mri_dmri=mri_dmri[mri_dmri['iqc_dmri_good_ser']>=1]

    #data=mri
    mri.info()

    if get == 'smri':
        r_data = mri_smri
        brain_features = [i for i in brain_features if 'smri' in i]
    elif get == 'nback':
        r_data = mri_nback
        brain_features = [i for i in brain_features if "tfmri_nback" in i]
    elif get == 'dmri':
        r_data = mri_dmri
        brain_features = [i for i in brain_features if "dmri" in i]
    elif get == 'sst':
        r_data = mri_sst
        brain_features = [i for i in brain_features if "tfmri_sst" in i]
    elif get == 'mid':
        r_data = mri_mid
        brain_features = [i for i in brain_features if "tfmri_mid" in i]
    elif get == 'rs':
        r_data = mri_rs
        brain_features = [i for i in brain_features if "rsfmri" in i]
    elif get == 'apriori':
        r_data = mri
        with open('apriori_brain_features.txt') as f:
            brain_features = f.read().splitlines()
    elif get == 'all':
        r_data = mri
    elif get == 'all_noqc':
        r_data = data
    else:
        r_data = None
        brain_features = None

    return r_data, targets, brain_features, behav_features, behav_dtypes

def get_setup_ML(get='nback',test_subjects='test_subjects',val_check=3):
    
    # Load necc. data
    data, targets, brain_features, behav_features, behav_dtypes = load_data(get,val_check)

    # Init
    ML = BPt_ML(log_dr=None)

    # Set load params
    ML.Set_Default_Load_Params(dataset_type='custom',
                               subject_id='src_subject_id',
                               eventname='baseline_year_1_arm_1',
                               overlap_subjects=False,
                               drop_na=False)
    #brain data
    ML.Load_Data(df=data,
                 inclusion_keys=brain_features,
                 filter_outlier_std=8)

    #target
    ML.Load_Targets(df=data,
                    col_name=targets,
                    data_type='f')

    #behavioral variables
    ML.Load_Covars(df=data,
                   col_name=behav_features,
                   data_type=behav_dtypes)

    # Strat
    ML.Load_Strat(df=data,
                  col_name='rel_family_id')

    # Define validation strategy as preserve by groups
    ML.Define_Validation_Strategy(groups='rel_family_id')

    # Define train test split
    if test_subjects is None:
        ML.Train_Test_Split(test_size=0.2,
                            random_state=1001)
    else:
        ML.Train_Test_Split(test_subjects=test_subjects)

    ML.Set_Default_ML_Verbosity(progress_bar='default', 
                                compute_train_score=False,
                                show_init_params='default', 
                                fold_name='default',
                                time_per_fold=True,
                                score_per_fold=True, 
                                fold_sizes=True, 
                                best_params=True)

    return ML