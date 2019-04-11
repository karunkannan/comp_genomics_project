import pandas as pd
from matplotlib import pyplot as plt
import os


def get_input():

    mut_filename = os.path.join('comp_genomics_project', '../data/TCGA_BRCA_loc_mutf.csv')
    mut_filename = os.path.abspath(os.path.realpath(mut_filename))

    mut_file = open(mut_filename)
    mut_df = pd.read_csv(mut_file)

    return mut_df


def pat_count_hist(mut_df):
    sum_list = list(mut_df)
    sum_list.remove('age_at_diagnosis')
    patient_total = mut_df[sum_list].sum(axis=1)
    plt.hist(patient_total,bins=patient_total.max())
    plt.xlabel("Number of Mutations")
    plt.ylabel("Number of Patients")
    plt.title("Histogram of Total Mutations per Patient")
    plt.figure()

    fig_filename = os.path.join('comp_genomics_project', '../plots/mut_per_patient_hist.png')
    fig_filename = os.path.abspath(os.path.realpath(fig_filename))
    plt.savefig(fig_filename)


def mut_count_hist(mut_df):
    col_list = list(mut_df)
    clin_col = ['age_at_diagnosis','race','gender', 'vital_status', 'primary_site', 'ethnicity', 'submitter_id', 'case_id']
    sum_list = [col for col in col_list if col not in clin_col]
    mut_total = mut_df[sum_list].sum(axis=0, skipna=True)
    plt.hist(mut_total, bins=mut_total.max())
    plt.xlabel("Number of Patients")
    plt.ylabel("Number of Mutations")
    plt.title("Histogram of Total Patients per Mutation")
    plt.figure()

    fig_filename = os.path.join('comp_genomics_project', '../plots/patient_per_mut_hist.png')
    fig_filename = os.path.abspath(os.path.realpath(fig_filename))
    plt.savefig(fig_filename)


def main():

    mut_df = get_input()
    pat_count_hist(mut_df)
    mut_count_hist(mut_df)
    plt.show()

    return 0


if __name__ == "__main__":
    main()
