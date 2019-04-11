import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
import statsmodels.stats.multitest as smm
from matplotlib import pyplot as plt


def get_input():

    mut_filename = os.path.join('comp_genomics_project', '../data/TCGA_BRCA_loc_mutf.csv')
    mut_filename = os.path.abspath(os.path.realpath(mut_filename))

    mut_file = open(mut_filename)
    mut_df = pd.read_csv(mut_file)

    return mut_df


def minor_allele_frequency(gene):
    allele_1 = 0
    allele_2 = 0
    for sample in gene:
        if sample == 0:
           allele_1 += 2
        elif sample == 1:
            allele_1 += 1
            allele_2 += 1
        elif sample == 2:
            allele_2 += 2
    if allele_1 > allele_2:
        maf = allele_2 / (allele_1 + allele_2)
    else:
        maf = allele_1 / (allele_1 + allele_2)

    return maf


def log_reg(mut_df):
    mut_df['survival'] = np.where(mut_df['vital_status'] == 'alive', 1, 0)
    mut_df['intercept'] = 1

    clin_col = ['age_at_diagnosis', 'race', 'gender', 'vital_status', 'primary_site', 'ethnicity', 'submitter_id',
                'case_id','survival','intercept']
    snp_list = [col for col in list(mut_df) if col not in clin_col]

    p_vals = []
    maf_list = []
    for snp in snp_list:
        maf = minor_allele_frequency(mut_df[snp])
        maf_list.append(maf)
        if maf < 0.005:
            continue
        else:
            log_model = sm.Logit(mut_df['survival'], mut_df[[snp,'intercept']])
            results = log_model.fit(method='bfgs', disp=0)
            p_vals.append(results.llr_pvalue)
    maf_list.sort()
    print(maf_list)
    return p_vals


def plot_pvals(p_values):
    num = range(0,len(p_values))
    plt.scatter(num,p_values)
    plt.xlabel("Test Number")
    plt.ylabel("Uncorrected P-value")
    plt.title("Uncorrected P-values for Vital Status of each SNP")
    plt.show()


def main():
    mut_df = get_input()
    p_vals = log_reg(mut_df)
    p_vals.sort()
    plot_pvals(p_vals)

    return 0


if __name__ == "__main__":
    main()
