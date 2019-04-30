import pandas as pd
import json
import argparse
from sklearn.preprocessing import OneHotEncoder
import numpy as np


def get_input(directory):
    """Open raw somatic MAF file and clinical JSON and return dataframes with relevant data"""

    use_col = ['Hugo_Symbol', 'Entrez_Gene_Id', 'Chromosome', 'Start_Position', 'End_Position', 'Variant_Type',
               'Reference_Allele', 'Tumor_Seq_Allele2', 'Gene', 'case_id']

    raw_maf_file = open(directory + "/TCGA.BRCA.varscan.6c93f518-1956-4435-9806-37185266d248.DR-10.0.somatic.maf")
    maf_df = pd.read_csv(raw_maf_file, sep='\t', skiprows=5, usecols=use_col)

    with open(directory + "/cases.2019-04-09.json") as clinical_file:
        clinical_data = json.load(clinical_file)

    maf_df['location'] = maf_df['Chromosome'] + maf_df['Start_Position'].astype(str)

    # Process Clinical JSON into a list for each case id
    clin_list = []
    for c_data in clinical_data:
        if 'diagnoses' not in c_data:
            continue
        clin_dict = {'case_id': c_data['case_id'], 'vital_status': c_data['diagnoses'][0]['vital_status'],
                     'age_at_diagnosis': c_data['diagnoses'][0]['age_at_diagnosis'],
                     'gender': c_data['demographic']['gender'], 'race': c_data['demographic']['race'],
                     'ethnicity': c_data['demographic']['ethnicity'], 'primary_site': c_data['primary_site'],
                     'submitter_id': c_data['submitter_id']}

        clin_list.append(clin_dict)

    # Turn the clinical list into a dataframe
    clin_df = pd.DataFrame(clin_list)
    clin_df.set_index('case_id', inplace=True)

    return maf_df, clin_df


def create_mutation_df(maf_df, case_ids):
    """Convert MAF file format into dataframe with indexes of patients and columns of locations"""

    gene_ids = maf_df['location'].unique()

    case_list = []
    for case in case_ids:
        case_dict = {'case_id': case}
        mutation_list = maf_df.loc[(maf_df['case_id'] == case), 'location'].tolist()
        for mutation in mutation_list:
            case_dict[mutation] = 1

        for gene in (set(gene_ids) - set(mutation_list)):
            case_dict[gene] = 0

        case_list.append(case_dict)

    mut_df = pd.DataFrame(case_list)
    mut_df.set_index('case_id', inplace=True)

    return mut_df


def filter_df(maf_df):
    """Filter out SNPs with only a single occurrence"""
    counts = maf_df['location'].value_counts()
    multi_counts = list(counts[counts > 1].index)
    multi_df = maf_df[maf_df['location'].isin(multi_counts)]

    return multi_df


def create_filtered_file(directory):
    """Create CSV file with filtered SNPs Dataframe"""
    maf_df, clc_df = get_input(directory)
    case_ids = maf_df['case_id'].unique()

    multi_df = filter_df(maf_df)
    mut_df = create_mutation_df(multi_df, case_ids)

    csv_df = pd.merge(mut_df, clc_df, on='case_id', how='inner')

    csv_df.to_csv(directory + "/TCGA_BRCA_loc_mutf.csv", sep=',')

    return 0


def create_full_file(directory):
    """Create CSV file with entire SNPs Data"""
    maf_df, clc_df = get_input(directory)
    case_ids = maf_df['case_id'].unique()

    mut_df = create_mutation_df(maf_df, case_ids)

    csv_df = pd.merge(mut_df, clc_df, on='case_id', how='inner')

    csv_df.to_csv(directory + "/TCGA_BRCA_loc_mut.csv", sep=',')

    return 0


def process_clinical_data(directory):
    """Process Clinical text file, including encoding and saving file"""
    clin_fp = open(directory + '/nationwidechildrens.org_clinical_patient_brca.txt')
    clin_data = pd.read_csv(clin_fp, skiprows=[1, 2], header=[0], delimiter="\t")
    keep_cols = ['bcr_patient_uuid', 'tumor_status', 'surgical_procedure_first', 'margin_status',
                 'lymph_nodes_examined_he_count', 'ajcc_tumor_pathologic_pt', 'ajcc_nodes_pathologic_pn',
                 'ajcc_metastasis_pathologic_pm', 'ajcc_pathologic_tumor_stage', 'er_status_by_ihc',
                 'pr_status_by_ihc', 'her2_status_by_ihc', 'histological_type', 'vital_status']

    clin_data = clin_data[keep_cols]

    clin_data['surgical_procedure_first'] = clin_data['surgical_procedure_first'].replace(
        ['[Discrepancy]', '[Unknown]', '[Not Available]'], 'NA_proc')
    first_surg_hot = OneHotEncoder(dtype=np.int, categories='auto')
    first_surg_out = first_surg_hot.fit_transform(clin_data[['surgical_procedure_first']]).toarray()
    clin_data = clin_data.merge(pd.DataFrame(first_surg_out, columns=np.array(first_surg_hot.categories_).ravel()),
                                left_index=True, right_index=True)

    clin_data['margin_status'] = clin_data['margin_status'].replace('[Unknown]', '[Not Available]')
    margin_map = {'[Not Available]': 0, 'Positive': 1, 'Close': 2, 'Negative': 3}
    clin_data['margin_status'] = clin_data['margin_status'].map(lambda x: margin_map[x])

    clin_data['tumor_status'] = clin_data['tumor_status'].replace('[Unknown]', '[Not Available]')
    tumor_map = {'[Not Available]': 0, 'WITH TUMOR': 1, 'TUMOR FREE': 2}
    clin_data['tumor_status'] = clin_data['tumor_status'].map(lambda x: tumor_map[x])

    clin_data['lymph_nodes_examined_he_count'] = clin_data['lymph_nodes_examined_he_count'].replace('[Not Available]', '-1')
    clin_data['lymph_nodes_examined_he_count'] = clin_data['lymph_nodes_examined_he_count'].astype(int)

    tumor_pt_map = {'TX': 0, 'T1': 1, 'T1a': 2, 'T1b': 3, 'T1c': 4, 'T2': 5, 'T2a': 6, 'T2b': 7, 'T3': 8, 'T3a': 9,
                    'T4': 10, 'T4b': 11, 'T4d': 12}
    clin_data['ajcc_tumor_pathologic_pt'] = clin_data['ajcc_tumor_pathologic_pt'].map(lambda x: tumor_pt_map[x])


    nodes_pn_map = {'NX': 0, 'N0 (mol+)': 1, 'N0 (i-)': 2, 'N0 (i+)': 3, 'N0': 4, 'N1mi': 5, 'N1': 6, 'N1a': 7,
                    'N1b': 8, 'N1c': 9, 'N2': 10, 'N2a': 11, 'N3': 12, 'N3a': 13, 'N3b': 14, 'N3c': 15}
    clin_data['ajcc_nodes_pathologic_pn'] = clin_data['ajcc_nodes_pathologic_pn'].map(lambda x: nodes_pn_map[x])

    meta_pm_map = {'MX': 0, 'M0': 1, 'cM0 (i+)': 2, 'M1': 3}
    clin_data['ajcc_metastasis_pathologic_pm'] = clin_data['ajcc_metastasis_pathologic_pm'].map(
        lambda x: meta_pm_map[x])

    clin_data['ajcc_pathologic_tumor_stage'] = clin_data['ajcc_pathologic_tumor_stage'].replace(
        ['[Not Available]', '[Discrepancy]'], 'Stage X')
    tumor_stage_map = {'Stage X': 0, 'Stage I': 1, 'Stage IA': 2, 'Stage IB': 3, 'Stage II': 4, 'Stage IIA': 5,
                       'Stage IIB': 6, 'Stage IIB': 7, 'Stage III': 8, 'Stage IIIA': 9, 'Stage IIIB': 10,
                       'Stage IIIC': 11, 'Stage IV': 12}
    clin_data['ajcc_pathologic_tumor_stage'] = clin_data['ajcc_pathologic_tumor_stage'].map(
        lambda x: tumor_stage_map[x])

    clin_data['er_status_by_ihc'] = clin_data['er_status_by_ihc'].replace('[Not Evaluated]', 'Indeterminate')
    er_status_map = {'Indeterminate': 0, 'Negative': 1, 'Positive': 2, }
    clin_data['er_status_by_ihc'] = clin_data['er_status_by_ihc'].map(lambda x: er_status_map[x])

    clin_data['pr_status_by_ihc'] = clin_data['pr_status_by_ihc'].replace('[Not Evaluated]', 'Indeterminate')
    pr_status_map = {'Indeterminate': 0, 'Negative': 1, 'Positive': 2, }
    clin_data['pr_status_by_ihc'] = clin_data['pr_status_by_ihc'].map(lambda x: pr_status_map[x])

    clin_data['her2_status_by_ihc'] = clin_data['her2_status_by_ihc'].replace(['[Not Available]', '[Not Evaluated]', 'Equivocal'], 'Indeterminate')
    her2_status_map = {'Indeterminate':0,'Negative': 1,'Positive':2,}
    clin_data['her2_status_by_ihc'] = clin_data['her2_status_by_ihc'].map(lambda x: her2_status_map[x])

    clin_data['histological_type'] = clin_data['histological_type'].replace(
        ['Other, specify', 'Mixed Histology (please specify)', '[Not Available]'], 'NA_hist')
    histo_type_hot = OneHotEncoder(dtype=np.int, categories='auto')
    histo_type_out = histo_type_hot.fit_transform(clin_data[['histological_type']]).toarray()
    clin_data = clin_data.merge(pd.DataFrame(histo_type_out, columns=np.array(histo_type_hot.categories_).ravel()),
                                left_index=True, right_index=True)

    clin_data.to_csv(directory + "/clinical_data_processed.csv", sep=',')
    return 1


def main():
    parser = argparse.ArgumentParser(description="Process Command line variables")
    parser.add_argument("--data_dir", nargs=1)
    args = parser.parse_args()
    directory = args.data_dir[0]
    create_filtered_file(directory)
    create_full_file(directory)
    process_clinical_data(directory)
    return 0


if __name__ == "__main__":
    main()
