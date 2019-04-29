import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import VarianceThreshold

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


def lin_svm_optimize(x_train, y_train, C=1):
    """ Create an SVM model with cross validation from the data and given c value"""

    lin_svm = svm.SVC(kernel='linear', C=C, class_weight='balanced', random_state=1)
    lin_scores = cross_val_score(lin_svm, x_train, y_train, cv=5)
    lin_avg = sum(lin_scores) / len(lin_scores)
    return lin_avg


def predict_lin_svm(x_train, y_train, x_test, y_test, c):
    """Create a score a linear svm model from the given data and c value"""

    # Create model
    lin_svm = svm.SVC(kernel='linear', C=c, class_weight='balanced', random_state=4)
    model = lin_svm.fit(x_train, y_train)

    # Score and predict
    score = model.score(x_test, y_test)
    print(score)
    y_predict = model.predict(x_test)
    return y_predict


def main():
    # Read in the genomic Data
    data = pd.read_csv('./data/TCGA_BRCA_loc_mutf.csv', index_col = 0)

    # For my data I didn't use most of these columns which is why I drop them in a couple lines.
    # You will probably want some of them to sort by
    clin_cols = ['age_at_diagnosis', 'ethnicity', 'gender', 'primary_site', 'race', 'submitter_id', 'vital_status']
    vital_encode = np.where(data['vital_status'] == 'alive', True, False)
    data = data.drop(clin_cols, axis=1)


    # Read in the clinical data
    clin_data = pd.read_csv('./data/clinical_data_processed.csv', index_col =0)
    clin_data['vital_status'] = np.where(clin_data['vital_status'] == 'Alive', True, False)

    # These columns are one hot encoded so ignore the categorical columns
    clin_exclude_cols = ['surgical_procedure_first', 'histological_type']
    clin_comb = clin_data[clin_data.columns.difference(clin_exclude_cols)]

    # Merge the genomic data and the clinical data on the patient ID
    full_data = data.merge(clin_comb, left_index=True, right_on='bcr_patient_uuid')
    full_data.set_index('bcr_patient_uuid',inplace=True,drop=True)

    # Split the vital status into its own datafram
    f_vital = full_data['vital_status']
    full_data.drop('vital_status',inplace=True, axis=1)

    # Train-test split the full data
    full_train, full_test, fvital_train, fvital_test = train_test_split(full_data,f_vital, test_size=0.3, random_state=1)

    # Drop columns with too low of a variance
    var_thresh = VarianceThreshold(.999 * (1 - .999))
    full_train_trim = var_thresh.fit_transform(full_train)
    full_test_trim = var_thresh.transform(full_test)

    # Drop to the best 10 features to speed up later feature selection
    selector_10 = SelectKBest(f_classif, k=10)
    full_train_10 = selector_10.fit_transform(full_train_trim, fvital_train)
    full_test_10 = selector_10.transform(full_test_trim)

    lin_svm_optimize(full_train_10, fvital_train)

    predict_lin_svm(full_train_10, fvital_train,full_test_10, fvital_test, 1)
    return 0


if __name__ == "__main__":
    main()

