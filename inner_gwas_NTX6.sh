#!/bin/sh

#################
# Inner GWAS
#################
echo "Inner GWAS starting"
# Change permissions for new files: Give other group members access:
umask uga+rwx

### Load modules
module load perl/5.20.1
module load plink2/1.90beta5.4
module load intel/perflibs/64
module load gcc/8.2.0
module load R/3.6.1
module load anaconda3/4.4.0

### Define path/variables
# The working/intermediate directory
export basepath='/home/projects/pr_46457/people/sarga/neurotox/data/intermediate'
# QC result file
export data='/home/projects/pr_46457/people/s174604/ML/RF/inner_gwas/QC_data'
# Path to pheno file
export pheno_oto="/home/projects/pr_46457/people/sarga/neurotox/data"

cd ${basepath}

### Association Analysis ###
# With covar 2,3 (cisplatin, age_atain)
plink --bfile ${data}'/A3302_final' --logistic hide-covar --pheno ${pheno_oto}'/GWAS_A3302_oto_binary_NTX6' --1 --covar ${data}'/cov.txt' --keep ${basepath}/intermidiate_train_ids_$1 --covar-number 2,3 --pfilter 0.0001 --out ${basepath}/inner_gwas_intermediate_$1
#plink --bfile ${data}'/A3302_final' --logistic hide-covar --all-pheno --pheno ${pheno_oto}'/GWAS_A3302_oto_continuous_NTX6' --covar ${data}'/cov.txt' --keep ${basepath}/intermidiate_train_ids --covar-number 2,3 --pfilter 0.00001 --out ${basepath}/inner_gwas_intermediate
#plink --bfile ${data}'/A3302_final' --assoc fisher --pheno ${pheno_oto}'/GWAS_A3302_oto_binary_NTX6' --1 --keep ${basepath}/intermidiate_train_ids --pfilter 0.0001 --out ${basepath}/inner_gwas_intermediate
#plink --bfile ${data}'/A3302_final' --assoc --pheno ${pheno_oto}'/GWAS_A3302_oto_binary_NTX6' --1 --keep ${basepath}/intermidiate_train_ids --pfilter 0.0001 --out ${basepath}/inner_gwas_intermediate

#sort -g -k9,9 ${basepath}/inner_gwas_intermediate.assoc.logistic > ${basepath}/inner_gwas_intermediate.assoc.logistic_sorted

# Exctract snp ids to file
tail -n +2 ${basepath}/inner_gwas_intermediate_$1.assoc.logistic | sed s/^\ *//g | tr -s ' ' ' ' | cut -f 2 -d ' ' > ${basepath}/intermediate_snps_$1.txt
#tail -n +2 ${basepath}/inner_gwas_intermediate.assoc.fisher | sed s/^\ *//g | tr -s ' ' ' ' | cut -f 2 -d ' ' > ${basepath}/intermediate_snps.txt
#tail -n +2 ${basepath}/inner_gwas_intermediate.assoc | sed s/^\ *//g | tr -s ' ' ' ' | cut -f 2 -d ' ' > ${basepath}/intermediate_snps.txt

echo "Inner GWAS complete"
