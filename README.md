# ABCD-DRD-Prediction

<p align="center">
  <img width="400" src="https://raw.githubusercontent.com/sahahn/Parcs_Project/master/data/abcd-study-logo.png">
</p>

This page contains code for the analyses from the study "One Year Predictions of Delayed Reward Discounting in the Adolescent Brain Cognitive Development Study". 

## Repository Structure

A brief description of the files provided in this repository and information on how they are used in the analyses are provided below:

- DRD_filter_dest.R Pull relevant data from ABCD RDS file
- loading_final.py: Data loading and processing code that covers all the steps from loading data as downloaded to their final preprocessed BPt-style ready for ML modelling.
- loading_final_site.py: Data loading and processing code that covers all the steps from loading data as downloaded to their final preprocessed BPt-style ready for ML modelling, alternative approach to participant grouping in cross-validation
- calculate_drd_curve.py: Code to compute the measure of delayed reward discounting predicted in this analyses
- run_analyses.py: run ML modeling analyses using BPt
- visualize_results.py: create figures and tables reported in manuscript using results of "run_analyses.py"

-drd_brain_features.txt: list of brain features used in analyses
-es_varlist_622.txt: list of non-brain features used in analyses
-drd_columns_final.csv: list of features and their data type used in BPt
-indif_varnames.txt: list of variables used to create DRD target variables
-factor_list.csv: list of which variables are factors for use in "DRD_filter_dest.R"

## Dependencies

- python3.6+
- R version 3.6.1
- brain-pred-toolbox version 1.3.4
- Anything else?
  

## Note on Reproducibility 

To run these scripts from start to finish, there are several required files that cannot be posted due to The Adolescent Brain Cognitive Development Study data security guidelines. For one, these scripts draw from the ABCD RDS file, which is based on ABCD release 2.0.1 and can be downloaded from the ABCD NDA portal (https://nda.nih.gov/abcd) by parties with data access. Additionally, for data filtering, files are needed listing participants of various classes, which cannot be posted for data security reasons. These files include lists of participants with acceptable fMRI data quality for the Nback (nback_good.txt), Stop Signal Task (sst_good.txt), and Monetary Incentive Delay Task (mid_good.txt), as well as lists of subjects to exclude due to having data from Philips scanners (philips.txt) or for having a glitch in their stop signal data (sst_exclude.txt). Further, a list of subjects in the lockbox test set (final_test_subs.txt) is needed for the cross-validation setup. We apologize for this inconvenience.

## Authors

Max M. Owens, Sage Hahn, Nicholas Allgaier, James MacKillop, Matthew Albaugh, Dekang Yuan, Anthony Juliano, Alexandra Potter, and Hugh Garavan.

## Contact

For questions, contact Max Owens at owensmax03@gmail.com.


<p align="center">
  <img width="600" src="https://raw.githubusercontent.com/sahahn/Parcs_Project/master/data/t32_logo.png">
</p>
