import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as rf
import matplotlib.pyplot as plt
import pickle

def train(X_train, Y_train, X_test, Y_test, fig_name):
    """
    X: Feature set
    Y: Labels
    """
    #n_estimators
    param = np.arange(1, 101)
    scores = []
    for i in param:
        clf = rf(n_estimators=i)
        clf.fit(X_train, Y_train)
        s = clf.score(X_test, Y_test)
        scores.append(s)
    n_estimators = (param[int(np.argmax(scores))], max(scores))
    plt.figure()
    plt.scatter(param, scores)
    plt.xlabel('# of Estimators')
    plt.ylabel('Accuracy')
    plt.title('Random Forest Accuracy over n_estimators')
    plt.savefig('{}_n_estimators'.format(fig_name))

    #max_depth
    param = np.arange(1, 101)
    scores = []
    for i in param:
        clf = rf(max_depth=i)
        clf.fit(X_train, Y_train)
        s = clf.score(X_test, Y_test)
        scores.append(s)
    max_depth = (param[int(np.argmax(scores))], max(scores))
    plt.figure()
    plt.scatter(param, scores)
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title('Random Forest Accuracy over Max Depth')
    plt.savefig('{}_max_depth'.format(fig_name))

    #min_samples_leaf
    param = np.arange(1, 101)
    scores = []
    for i in param:
        clf = rf(n_estimators=i)
        clf.fit(X_train, Y_train)
        s = clf.score(X_test, Y_test)
        scores.append(s)
    min_samples_leaf = (param[int(np.argmax(scores))], max(scores))
    plt.figure()
    plt.scatter(param, scores)
    plt.xlabel('Min. Samples per Leaf')
    plt.ylabel('Accuracy')
    plt.title('Random Forest Accuracy over Min Samples per Leaf')
    plt.savefig('{}_min_samples_leaf'.format(fig_name))

    v = [n_estimators[1], max_depth[1], min_samples_leaf[1]]
    idx = np.argmax(v)
    if idx == 0:
        clf = rf(n_estimators=n_estimators[0])
    elif idx == 1:
        clf = rf(max_depth=max_depth[0])
    else:
        clf = rf(min_samples_leaf=min_samples_leaf[0])
    clf.fit(X_train, Y_train)
    return clf

def read_data(data_dir):
    snp_data = pd.read_csv('{}/TCGA_BRCA_loc_mutf.csv'.format(data_dir), index_col=0)
    snp_data.drop(['primary_site', 'submitter_id'],
            inplace=True, axis=1)
    clinical_data = pd.read_csv('{}clinical_data_processed.csv'.format(data_dir),
            index_col=1)
    clinical_data = clinical_data.iloc[:, 1:]
    clinical_data.drop(['surgical_procedure_first', 'vital_status',
        'histological_type'],
            inplace=True, axis=1)

    encoding_map = {'not hispanic or latino': 0, 'hispanic or latino': 1,
    'black or african american':0, 'white':1,'asian': 2,
    'american indian or alaska native':3, 'not reported': -1, 'female': 0,
    'male': 1, 'alive': 1, 'dead': 0}
    col2mod = [991, 992, 990, 993]
    for c in col2mod:
        num_samples = snp_data.shape[0]
        for i in range(num_samples):
            val = snp_data.iloc[i, c ]
            snp_data.iloc[i, c ] = encoding_map[val]

    snp_data.dropna(axis='index', how='any', inplace=True)
    clinical_data.dropna(axis='index', how='any', inplace=True)

    return snp_data, clinical_data

def acc_score(clf, X_test, Y_test, out):
    pred = clf.predict(X_test)
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(len(Y_test)):
        if pred[i] == Y_test[i]:
            if pred[i]:
                tp += 1
            else:
                tn += 1
        else:
            if pred[i]:
                fp += 1
            else:
                fn += 1

    score = clf.score(X_test, Y_test)

    with open(out, 'w') as f:
        f.write(', P, N\n')
        f.write('P, {}, {}\n'.format(tp, fp))
        f.write('N, {}, {}\n'.format(fn, tn))
        f.write('Score: {}'.format(score))

    return tp, fp, fn, tn

def train_variance_filter(data_dir, results_dir):
    data_dir = '{}/'.format(data_dir)
    results_dir = '{}/'.format(results_dir)
    snp, clinical = read_data(data_dir)
    all_data = snp.join(clinical, how='inner')

    idx = all_data.columns.get_loc("age_at_diagnosis")
    max_val = max(all_data['age_at_diagnosis'].values)
    for i in range(all_data.shape[0]):
        all_data.iloc[i, idx] = all_data.iloc[i, idx]/max_val

    labels = all_data['vital_status'].values
    all_data.drop(['vital_status'], inplace=True, axis=1)
    variance = all_data.var()
    names = variance.index
    mean_var = np.mean(variance.values)
    keep = []

    for i in range(len(variance)):
        if variance.iloc[i] > mean_var - 0.02:
            keep.append(i)
    kept_data = all_data.iloc[:, keep]

    X_train, X_test, Y_train, Y_test = train_test_split(kept_data.values, labels,
            test_size=0.3)

    clf = train(X_train, Y_train, X_test, Y_test, '{}rf_var_filter'.format(results_dir))
    with open('{}rf_var_filter.pkl'.format(results_dir), 'wb') as f:
        pickle.dump(clf, f)
    with open('{}rf_var_filter_X_test.pkl'.format(results_dir), 'wb') as f:
        pickle.dump(X_test, f)
    with open('{}rf_var_filter_Y_test.pkl'.format(results_dir), 'wb') as f:
        pickle.dump(Y_test, f)
    #acc_score(clf, X_test, Y_test, '{}var_filter_score.txt'.format(results_dir))

def test_variance_filter(data_dir, results_dir):
    data_dir = '{}/'.format(data_dir)
    results_dir = '{}/'.format(results_dir)
    with open('{}rf_var_filter.pkl'.format(results_dir), 'rb') as f:
        clf = pickle.load(f)
    with open('{}rf_var_filter_X_test.pkl'.format(results_dir), 'rb') as f:
        X_test = pickle.load(f)
    with open('{}rf_var_filter_Y_test.pkl'.format(results_dir), 'rb') as f:
        Y_test = pickle.load(f)
    acc_score(clf, X_test, Y_test, '{}rf_var_filter_score.txt'.format(results_dir))

def train_all_data(data_dir, results_dir):
    data_dir = '{}/'.format(data_dir)
    results_dir = '{}/'.format(results_dir)
    snp, clinical = read_data(data_dir)
    all_data = snp.join(clinical, how='inner')
    labels = all_data['vital_status'].values
    all_data.drop(['vital_status'], inplace=True, axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(all_data.values, labels,
            test_size=0.3)

    clf = train(X_train, Y_train, X_test, Y_test, '{}rf_all'.format(results_dir)) 
    #acc_score(clf, X_test, Y_test, '{}all_data_score.txt'.format(results_dir))
    with open('{}rf_all_data.pkl'.format(results_dir), 'wb') as f:
        pickle.dump(clf, f)
    with open('{}rf_all_data_X_test.pkl'.format(results_dir), 'wb') as f:
        pickle.dump(X_test, f)
    with open('{}rf_all_data_Y_test.pkl'.format(results_dir), 'wb') as f:
        pickle.dump(Y_test, f)

def test_variance_filter(data_dir, results_dir): 
    data_dir = '{}/'.format(data_dir)
    results_dir = '{}/'.format(results_dir)
    with open('{}rf_all_data.pkl'.format(results_dir), 'rb') as f:
        clf = pickle.load(f)
    with open('{}rf_all_data_X_test.pkl'.format(results_dir), 'rb') as f:
        X_test = pickle.load(f)
    with open('{}rf_all_data_Y_test.pkl'.format(results_dir), 'rb') as f:
        Y_test = pickle.load(f)
    acc_score(clf, X_test, Y_test, '{}rf_all_data_score.txt'.format(results_dir))

def train_corr_filter(data_dir, results_dir):
    data_dir = '{}/'.format(data_dir)
    results_dir = '{}/'.format(results_dir)
    snp, clinical = read_data(data_dir)
    all_data = snp.join(clinical, how='inner')

    corr = all_data.corr()
    corr = corr['vital_status']
    corr = corr.nlargest(n=101)
    #print(corr.index.values[1:])
    all_data = all_data[corr.index.values]

    labels = all_data['vital_status'].values
    all_data.drop(['vital_status'], inplace=True, axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(all_data.values, labels,
            test_size=0.3)

    clf = train(X_train, Y_train, X_test, Y_test, '{}rf_corr'.format(results_dir))
    #acc_score(clf, X_test, Y_test, '{}corr_data_score.txt'.format(results_dir)) 
    with open('{}rf_corr_filter.pkl'.format(results_dir), 'wb') as f:
        pickle.dump(clf, f)
    with open('{}rf_corr_filter_X_test.pkl'.format(results_dir), 'wb') as f:
        pickle.dump(X_test, f)
    with open('{}rf_corr_filter_Y_test.pkl'.format(results_dir), 'wb') as f:
        pickle.dump(Y_test, f)

def test_variance_filter(data_dir, results_dir): 
    data_dir = '{}/'.format(data_dir)
    results_dir = '{}/'.format(results_dir)
    with open('{}rf_corr_filter.pkl'.format(results_dir), 'rb') as f:
        clf = pickle.load(f)
    with open('{}rf_corr_filter_X_test.pkl'.format(results_dir), 'rb') as f:
        X_test = pickle.load(f)
    with open('{}rf_corr_filter_Y_test.pkl'.format(results_dir), 'rb') as f:
        Y_test = pickle.load(f)
    acc_score(clf, X_test, Y_test,
            '{}rf_corr_filter_score.txt'.format(results_dir))

def main():
    snp_data, clinical_data = read_data("data/")
    all_data = snp_data.join(clinical_data, how='inner')
    labels = all_data['vital_status'].values
    all_data.drop(['vital_status'], inplace=True, axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(all_data.values, labels,
            test_size=0.3)

    #print(all_data['Infiltrating Ductal Carcinoma'])

    train(X_train, Y_train, X_test, Y_test, 'plots/all')

if __name__=="__main__":
    main()

