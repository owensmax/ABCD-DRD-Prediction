# DEAP model 2017
###SET THESE BEFORE RUNNING#####
targets=readLines("/home/max/Documents/DRD/indif_varnames.txt")
behav_features=readLines("/home/max/Documents/DRD/es_varlist_622.txt")
brain_features=readLines("/home/max/Documents/DRD/drd_brain_features.txt")
strat_vars=c("rel_family_id","mri_info_device.serial.number")
qc_vars=c("iqc_rsfmri_good_ser",'iqc_dmri_good_ser','fsqc_qc','tfmri_mid_all_beta_mean.motion',
          'tfmri_nback_all_beta_mean.motion','tfmri_sst_all_beta_mean.motion',
          'iqc_rsfmri_all_mean_motion','rsfmri_var_ntpoints','iqc_dmri_all_mean_motion')
allvars=c('src_subject_id','eventname',qc_vars,strat_vars,behav_features,brain_features)
#data =  readRDS( paste0("/home/max/Documents/linear_mixed_model_abcd/nda2.0.1.Rds"))
#backup_data=data
data=backup_data
data_yr1=data[c('src_subject_id','eventname',targets)]
data_yr1=data_yr1[ which(data_yr1$eventname=='1_year_follow_up_y_arm_1'),]
data_yr1 <- data_yr1[c(1,3:9)]
data = data[allvars]
data <-data[ which(data$eventname=='baseline_year_1_arm_1'),]
data<-merge(data,data_yr1)

facs=readLines('/home/max/Documents/DRD/factor_list.csv')
data[facs] <- lapply(data[facs], as.numeric)
#data3<-data[facs_cols]

write.csv(data,"/home/max/Documents/DRD/drd_data_dest.csv")

data[behav_features,]

typeof(data$cbcl_scr_syn_anxdep_r)
typeof(data$src_subject_id)
typeof(data$sex)
