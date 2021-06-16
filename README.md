# ABCD-DRD-Prediction

This page contains code for the analyses from the study "One Year Predictions of Delayed Reward Discounting in the Adolescent Brain Cognitive Development Study". 

## Repository Structure

A brief description of the files provided in this repository and information on how they are used in the analyses are provided below:

- calculate_drd_curve.py: Code to compute the measure of delayed reward discounting predicted in this analyses.
- loading_final.py: Data loading and processing code that covers all the steps from loading data as downloaded to their final preprocessed BPt-style ready for ML modelling.
- ... 
- ...

* Suggestion: order them roughly in the order they are used *


## Dependencies

- python3.6+
- R (some version)
- brain-pred-toolbox==1.3.6 (Is this right???)
- Anything else?
  

## Note on Reproducibility 

To run these scripts from start to finish, there are several required files that cannot be posted due to The Adolescent Brain Cognitive Development Study 
data security guidelines. For one, these scripts draw from the ABCD RDS file, which is based on ABCD release 2.0.1 and can be downloaded from the ABCD NDA 
portal (https://nda.nih.gov/abcd) by parties with data access. Additionally, for data filtering, files are needed listing participants of various classes, 
which cannot be posted for data security reasons. These files include lists of participants with acceptable fMRI data quality for the Nback (nback_good.txt), 
Stop Signal Task (sst_good.txt), and Monetary Incentive Delay Task (mid_good.txt), as well as lists of subjects to exclude due to having data from Philips 
scanners (philips.txt) or for having a glitch in their stop signal data (sst_exclude.txt). Further, a list of subjects in the lockbox test set (final_test_subs.txt) 
is needed for the cross-validation setup. We apologize for this inconvenience.

## Authors

Max M. Owens, Sage Hahn, Nicholas Allgaier, James MacKillop, Matthew Albaugh, Dekang Yuan, Anthony Juliano, Alexandra Potter, and Hugh Garavan.

## Contact

For questions, contact Max Owens at owensmax03@gmail.com.
