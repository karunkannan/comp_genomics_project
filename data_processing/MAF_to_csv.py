import pandas as pd
import json
import os


def get_input():

    use_col=['Hugo_Symbol', 'Entrez_Gene_Id', 'Chromosome', 'Start_Position', 'End_Position', 'Variant_Type',
             'Reference_Allele', 'Tumor_Seq_Allele2', 'Gene', 'case_id']

    maf_filename = os.path.join('comp_genomics_project',
                                '../data/TCGA.BRCA.varscan.6c93f518-1956-4435-9806-37185266d248.DR-10.0.somatic.maf')
    maf_filename = os.path.abspath(os.path.realpath(maf_filename))

    raw_maf_file = open(maf_filename)
    maf_df = pd.read_csv(raw_maf_file, sep='\t', skiprows=5, usecols=use_col)


    clin_filename = os.path.join('comp_genomics_project',
                                 '../data/cases.2019-04-09.json')
    clin_filename = os.path.abspath(os.path.realpath(clin_filename))
    with open(clin_filename) as clinical_file:
        clinical_data = json.load(clinical_file)

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

    clin_df = pd.DataFrame(clin_list)
    clin_df.set_index('case_id', inplace=True)

    return maf_df, clin_df


def create_mutation_df(maf_df):
    gene_ids = maf_df['Gene'].unique()
    case_ids = maf_df['case_id'].unique()

    case_list = []
    for case in case_ids:
        case_dict = {'case_id': case}
        mutation_list = maf_df.loc[(maf_df['case_id'] == case), 'Gene'].tolist()
        for mutation in mutation_list:
            case_dict[mutation] = 1

        for gene in (set(gene_ids) - set(mutation_list)):
            case_dict[gene] = 0

        case_list.append(case_dict)

    mut_df = pd.DataFrame(case_list)
    mut_df.set_index('case_id', inplace=True)

    return mut_df


def main():
    maf_df, clc_df = get_input()
    mut_df = create_mutation_df(maf_df)

    csv_df = pd.merge(mut_df, clc_df, on='case_id', how='inner')

    csv_df.to_csv("./data/TCGA_BRCA_mutations.csv", sep=',')

    return 1


if __name__ == "__main__":
    main()
