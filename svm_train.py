import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.feature_selection import VarianceThreshold
from sklearn.externals import joblib


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
# Gaussian SVM generates many warnings that clog console
simplefilter(action='ignore', category=FutureWarning)


def read_data(directory):
    """Read in the data, remove the unused clinical columns, and enocde the vital status"""
    data = pd.read_csv(directory + '/TCGA_BRCA_loc_mutf.csv', index_col=0)

    clin_cols = ['age_at_diagnosis', 'ethnicity', 'gender', 'primary_site', 'race', 'submitter_id', 'vital_status']
    vital_encode = np.where(data['vital_status'] == 'alive', True, False)
    data = data.drop(clin_cols, axis=1)

    clin_data = pd.read_csv(directory + '/clinical_data_processed.csv', index_col=0)
    clin_data['vital_status'] = np.where(clin_data['vital_status'] == 'Alive', True, False)
    clin_exclude_cols = ['surgical_procedure_first', 'histological_type']
    clin_comb = clin_data[clin_data.columns.difference(clin_exclude_cols)]
    full_data = data.merge(clin_comb, left_index=True, right_on='bcr_patient_uuid')
    full_data.set_index('bcr_patient_uuid', inplace=True, drop=True)
    f_vital = full_data['vital_status']
    full_data.drop('vital_status', inplace=True, axis=1)

    return data, vital_encode, full_data, f_vital


def lin_svm_optimize(x_train, y_train, C=1):
    """Create linear svm model from the provided data and C"""
    lin_svm = svm.SVC(kernel='linear', C=C, class_weight='balanced', random_state=1)
    lin_scores = cross_val_score(lin_svm, x_train, y_train, cv=5)
    lin_avg = sum(lin_scores) / len(lin_scores)
    return lin_avg


def gauss_svm_optimize(x_train,y_train, C=1, logGamma=0):
    """Create linear svm model from the provided data and hyperparameters"""
    gauss_svm = svm.SVC(kernel='rbf', C=C, gamma=10**logGamma, class_weight='balanced')
    gauss_scores = cross_val_score(gauss_svm, x_train, y_train, cv=5)
    gauss_avg = sum(gauss_scores) / len(gauss_scores)
    return gauss_avg


def lin_svm_pipe(x_train, y_train, cmax, k, fig_dir, version):
    """Pipeline to find the best hyperparameters for linear svm, and return the best hyper parameters"""
    c_vals = range(1, cmax + 1)
    k_vals = range(1, k + 1)
    results = np.zeros((k, cmax))
    for k in k_vals:
        x_train_new = SelectKBest(f_classif, k=k).fit_transform(x_train, y_train)
        for c in c_vals:
            results[k - 1][c - 1] = lin_svm_optimize(x_train_new, y_train, c)

    plot_vals(k_vals, c_vals, results, "Number of features", "C Value",
              "Linear SVM Accuracy based on Features and C: " + version, 'lin_svm_train_' + version, fig_dir)

    k_best, c_best = np.unravel_index(results.argmax(), results.shape)
    k_best += 1
    c_best += 1
    return k_best, c_best


def gauss_svm_pipe(x_train, y_train, cmax, lGmin, k, fig_dir):
    """Pipeline to find the best hyperparameters for Gaussian svm, and return the best hyper parameters"""
    c_vals = range(1, cmax + 1)
    lG_vals = range(lGmin, 1, 1)
    results = np.zeros((len(lG_vals), cmax))
    for i, lG in enumerate(lG_vals):
        for c in c_vals:
            results[i][c - 1] = gauss_svm_optimize(x_train, y_train, c, lG)

    plot_vals(lG_vals, c_vals, results, "Log Gamma Value", "C Value",
              "Gaussian SVM Accuracy based on Log Gamma and C: " + str(k) + " Features", 'gauss_svm_train' + str(k), fig_dir)
    lG_best_index, c_best = np.unravel_index(results.argmax(), results.shape)
    lG_best = lG_vals[lG_best_index]
    c_best += 1
    return lG_best, c_best


def plot_vals(x, y, results, xlabel, ylabel, title, fp_name, directory):
    """Create 3d plot of the accuracy of the training models and save it out"""
    plt.clf()
    X, Y = np.meshgrid(x,y)
    Z = results.reshape(X.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel('Accuracy')
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(directory + "/" + fp_name + ".png")
    return None


def count_prediction(y_predict, y_test):
    """Count the number of true positive, true negatives, false positives, and false negatives"""
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(0,len(y_predict)):
        if y_predict[i] == y_test[i] and y_predict[i]:
            TP += 1
        elif y_predict[i] == y_test[i] and (not y_predict[i]):
            TN += 1
        elif y_predict[i] != y_test[i] and y_predict[i]:
            FP += 1
        else:
            FN += 1
    return TP, FP, FN, TN


def create_lin_model(x_train, y_train, k, c):
    """Create  linear SVM model from the provided training data and hyper parameters"""
    selector = SelectKBest(f_classif, k=k)
    x_train_new = selector.fit_transform(x_train, y_train)
    lin_svm = svm.SVC(kernel='linear', C=c, class_weight='balanced', random_state=4)
    model = lin_svm.fit(x_train_new, y_train)
    return model, selector


def create_gauss_model(x_train, y_train, k, c, lG):
    """Create Guassian SVM model from the provided training data and hyper parameters"""
    selector = SelectKBest(f_classif, k=k)
    x_train_new = selector.fit_transform(x_train, y_train)
    lin_svm = svm.SVC(kernel='rbf', C=c, gamma=10**lG, class_weight='balanced', random_state=4)
    model = lin_svm.fit(x_train_new, y_train)
    return model, selector


def predict_svm(model, selector, x_test, y_test):
    """Predict the y values using the given model and selector """
    x_test_trim = selector.transform(x_test)
    score = model.score(x_test_trim, y_test)
    y_predict = model.predict(x_test_trim)
    return score, count_prediction(y_predict, y_test)


def train_linear_svm(directory, results_directory):
    """Model Training for Linear SVM. Output the model parameters to the correct file"""
    data, vital_encode, full_data, f_vital = read_data(directory)

    # Filter and select features for the training
    snp_train, snp_test, vital_train, vital_test = train_test_split(data, vital_encode, test_size=0.3, random_state=1)

    # Remove low variance SNPs
    var_thresh = VarianceThreshold(.999 * (1 - .999))
    snp_train_trim = var_thresh.fit_transform(snp_train)

    # Select 50 best
    selector_50 = SelectKBest(f_classif, k=50)
    snp_train_50 = selector_50.fit_transform(snp_train_trim, vital_train)

    k_best, c_best = lin_svm_pipe(snp_train_50, vital_train, 10, 30, results_directory, 'SNP')
    lin_model, lin_select = create_lin_model(snp_train_50, vital_train, k_best, c_best)

    # Save out the best model
    joblib.dump(lin_model, results_directory + "/lin_svm_model.pkl")
    joblib.dump(lin_select, results_directory + "/lin_svm_selector.pkl")

    # Training for Clinical and SNP data
    full_train, full_test, fvital_train, fvital_test = train_test_split(full_data, f_vital, test_size=0.3,
                                                                        random_state=1)
    var_thresh = VarianceThreshold(.999 * (1 - .999))
    full_train_trim = var_thresh.fit_transform(full_train)
    selector_50 = SelectKBest(f_classif, k=50)
    full_train_50 = selector_50.fit_transform(full_train_trim, fvital_train)

    k_best, c_best = lin_svm_pipe(full_train_50, fvital_train, 10, 30, results_directory, 'Full')
    lin_fmodel, lin_fselect = create_lin_model(full_train_50, fvital_train, k_best, c_best)
    joblib.dump(lin_fmodel, results_directory + "/linf_svm_model.pkl")
    joblib.dump(lin_fselect, results_directory + "/linf_svm_selector.pkl")

    return 0


def train_gauss_svm(directory, results_directory):
    """Model Training for Gaussian SVM. Output the model parameters to the correct file"""
    data, vital_encode, full_data, f_vital = read_data(directory)

    # Filter and select features for the training
    full_train, full_test, fvital_train, fvital_test = train_test_split(full_data, f_vital, test_size=0.3,
                                                                        random_state=1)
    var_thresh = VarianceThreshold(.999 * (1 - .999))
    full_train_trim = var_thresh.fit_transform(full_train)
    selector_50 = SelectKBest(f_classif, k=5)
    full_train_50 = selector_50.fit_transform(full_train_trim, fvital_train)

    lG_best, c_best = gauss_svm_pipe(full_train_50, fvital_train, 10, -10, 5, results_directory)
    gauss_model, gauss_select = create_gauss_model(full_train_50, fvital_train, 5, c_best, lG_best)

    # Save out the best model
    joblib.dump(gauss_model, results_directory + "/gauss_svm_model.pkl")
    joblib.dump(gauss_select, results_directory + "/gauss_svm_selector.pkl")


def predict_linear_svm(data_directory, results_directory):
    """Predict for Linear SVM. Output the accuracy to the correct file"""
    data, vital_encode, full_data, f_vital = read_data(data_directory)

    # Filter and select features for the training
    snp_train, snp_test, vital_train, vital_test = train_test_split(data, vital_encode, test_size=0.3, random_state=1)

    var_thresh = VarianceThreshold(.999 * (1 - .999))
    snp_train_trim = var_thresh.fit_transform(snp_train)
    snp_test_trim = var_thresh.transform(snp_test)

    # Select 50 best
    selector_50 = SelectKBest(f_classif, k=50)
    snp_train_50 = selector_50.fit_transform(snp_train_trim, vital_train)
    snp_test_50 = selector_50.transform(snp_test_trim)

    lin_model = joblib.load(results_directory + "/lin_svm_model.pkl")
    lin_selector = joblib.load(results_directory + "/lin_svm_selector.pkl")

    score, counts = predict_svm(lin_model, lin_selector, snp_test_50, vital_test)

    with open(results_directory + "/lin_SVM_score.txt", 'w') as fp:
        fp.write("Linear Score\n")
        fp.write(str(score) + "\n")
        fp.write("TP, FP, FN, TN\n")
        fp.write(str(counts))


    # Predict on full data
    full_train, full_test, fvital_train, fvital_test = train_test_split(full_data, f_vital, test_size=0.3,
                                                                        random_state=1)
    var_thresh = VarianceThreshold(.999 * (1 - .999))
    full_train_trim = var_thresh.fit_transform(full_train)

    full_test_trim = var_thresh.transform(full_test)
    selector_50 = SelectKBest(f_classif, k=50)
    full_train_50 = selector_50.fit_transform(full_train_trim, fvital_train)
    full_test_50 = selector_50.transform(full_test_trim)

    linf_model = joblib.load(results_directory + "/linf_svm_model.pkl")
    linf_selector = joblib.load(results_directory + "/linf_svm_selector.pkl")

    score, counts = predict_svm(linf_model, linf_selector, full_test_50, fvital_test)

    with open(results_directory + "/linf_SVM_score.txt", 'w') as fp:
        fp.write("Linear Score\n")
        fp.write(str(score) + "\n")
        fp.write("TP, FP, FN, TN\n")
        fp.write(str(counts))

    return 0


def predict_gauss_svm(data_directory, results_directory):
    """Predict for Linear SVM. Output the accuracy to the correct file"""
    data, vital_encode, full_data, f_vital = read_data(data_directory)

    # Filter and select features for the training
    full_train, full_test, fvital_train, fvital_test = train_test_split(full_data, f_vital, test_size=0.3,
                                                                        random_state=1)
    var_thresh = VarianceThreshold(.999 * (1 - .999))
    full_train_trim = var_thresh.fit_transform(full_train)
    full_test_trim = var_thresh.transform(full_test)
    selector_50 = SelectKBest(f_classif, k=5)
    full_train_50 = selector_50.fit_transform(full_train_trim, fvital_train)
    full_test_50 = selector_50.transform(full_test_trim)

    gauss_model = joblib.load(results_directory + "/gauss_svm_model.pkl")
    gauss_selector = joblib.load(results_directory + "/gauss_svm_selector.pkl")

    score, counts = predict_svm(gauss_model, gauss_selector, full_test_50, fvital_test)

    with open(results_directory + "/gauss_SVM_score.txt", 'w') as fp:
        fp.write("Linear Score\n")
        fp.write(str(score) + "\n")
        fp.write("TP, FP, FN, TN\n")
        fp.write(str(counts))

    return 0


def main():
    predict_linear_svm("./data")
    return 0


if __name__ == "__main__":
    main()
