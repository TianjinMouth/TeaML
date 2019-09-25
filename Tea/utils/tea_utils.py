import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score


def feature_select(x_train, y_train, method='iv', kb=100, rfe=30):
    if method == 'iv':
        method = mutual_info_classif
    elif method == 'f':
        method = f_classif

    # chi2
    fn = x_train.columns
    selector1 = SelectKBest(chi2, kb)
    selector1.fit(x_train, y_train)

    # information value
    selector2 = SelectKBest(method, kb)
    selector2.fit(x_train, y_train)
    left_features = list(set(fn[selector2.get_support()].tolist() + fn[selector1.get_support()].tolist()))

    # RFE
    _X_tmp = x_train[left_features]
    fn = _X_tmp.columns
    clf = LogisticRegression(penalty='l2', C=0.2)
    selector = RFE(estimator=clf, n_features_to_select=rfe)
    selector.fit(_X_tmp, y_train)

    left_features = fn[selector.get_support()].tolist()
    x_train = x_train[left_features]
    return left_features


def compute_ks(prob, target):
    """
    target: numpy array of shape (1,)
    prob: numpy array of shape (1,), predicted probability of the sample being positive
    returns:
    ks: float, ks score estimation
    """
    get_ks = lambda prob, target: ks_2samp(prob[target == 1], prob[target != 1]).statistic

    return get_ks(prob, target)


def train_by_cv(x, y, x_oot, y_oot, sss, clf, weight=None, **kw):
    pbar = tqdm(total=100)
    auc_train, auc_test, auc_oot = [], [], []
    ks_train, ks_test, ks_oot = [], [], []
    stacking_train = []
    stacking_oot = []
    oos_idx = []
    for train_index, test_index in sss.split(x, y):
        x_train, x_test = x.iloc[train_index, :], x.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        if weight is not None:
            clf.fit(x_train, y_train, sample_weight=weight[train_index])
        else:
            clf.fit(x_train, y_train, **kw)
        oos_pred = clf.predict_proba(x_test)[:, 1]
        oot_pred = clf.predict_proba(x_oot)[:, 1]
        oos_idx.extend(test_index)
        stacking_train.extend(oos_pred)
        stacking_oot.append(oot_pred)
        auc_train.append(roc_auc_score(y_train, clf.predict_proba(x_train)[:, 1]))
        auc_test.append(roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1]))
        auc_oot.append(roc_auc_score(y_oot, clf.predict_proba(x_oot)[:, 1]))
        ks_train.append(compute_ks(clf.predict_proba(x_train)[:, 1], y_train))
        ks_test.append(compute_ks(clf.predict_proba(x_test)[:, 1], y_test))
        ks_oot.append(compute_ks(clf.predict_proba(x_oot)[:, 1], y_oot))
        pbar.update(20)

    pbar.close()
    stacking_train = pd.Series(stacking_train, index=oos_idx).sort_index().values
    stacking_oot = np.array(stacking_oot).mean(axis=0)
    print("Train AUC: %s" % np.mean(auc_train))
    print("Test AUC: %s" % np.mean(auc_test))
    print("OOT AUC: %s" % np.mean(auc_oot))
    print("Train KS: %s" % np.mean(ks_train))
    print("Test KS: %s" % np.mean(ks_test))
    print("OOT KS: %s" % np.mean(ks_oot))
    print("--------------------------------------------------- \n")

    return clf, stacking_train, stacking_oot


def get_importance(opt, x):
    try:
        feature_coef = pd.concat([
            pd.DataFrame(pd.DataFrame(x.columns, columns=['feature_name'])),
            pd.DataFrame(opt.coef_.T, columns=['feature_coef'])
        ],  axis=1)
    except AttributeError:
        feature_coef = pd.concat([
            pd.DataFrame(x.columns, columns=['feature_name']),
            pd.DataFrame(opt.feature_importances_.T, columns=['feature_coef'])
        ],  axis=1)
    feature_coef['abs'] = np.abs(feature_coef['feature_coef'])
    feature_coef = feature_coef.sort_values(by='abs', ascending=False)
    print(feature_coef.sort_values(by='abs', ascending=False)[['feature_name', 'feature_coef']].head(20))
    return feature_coef[['feature_name', 'feature_coef']]

