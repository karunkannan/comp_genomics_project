import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.multitest import fdrcorrection


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


def GWAS_plot(data_dir, results_dir):
    data = pd.read_csv(data_dir + '/TCGA_BRCA_loc_mut.csv', index_col=0)

    clin_cols = ['age_at_diagnosis', 'ethnicity', 'gender', 'primary_site', 'race', 'submitter_id', 'vital_status']

    plt.hist(data[data.columns.difference(clin_cols)].sum(axis=0), bins=range(0, 141, 10), log=True)
    plt.xlabel("Number of Patients with SNP")
    plt.ylabel("Number of Occurences (log-scale)")
    plt.title("Number of Patients with SNPs Histogram")
    plt.savefig(results_dir + "/SNP_histogram.png")

    data = pd.read_csv(data_dir + '/TCGA_BRCA_loc_mutf.csv', index_col=0)

    maf_snps = []
    for snp in data:
        if snp not in clin_cols and minor_allele_frequency(data[snp]) > 0.005:
            maf_snps.append(snp)

    data['intercept'] = 1
    data['vital_encode'] = np.where(data['vital_status'] == 'alive', 1, 0)

    p_vals = []
    for snp in maf_snps:
        log_model = sm.Logit(data['vital_encode'], data[[snp, 'intercept']])
        results = log_model.fit(method='bfgs', disp=0)
        p_vals.append(results.pvalues[0])

    bonferroni = 0.05 / len(maf_snps)
    p_vals.sort()

    plt.clf()
    plt.scatter(x=np.array(range(1, len(p_vals) + 1)), y=p_vals)
    plt.axhline(y=bonferroni, color='r', linestyle='--', label='Bonferroni (FWER)')
    plt.xlabel("SNP Number")
    plt.ylabel("p-vales")
    plt.legend()
    plt.title("Corrected p-values of logistic regression between SNPs and Vital Status")
    plt.savefig(results_dir + "/GWAS_pvals.png")
