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


def compute_pred_psi(train, oot):
    # 计算train和oot分数的psi
    cuts = np.arange(10, 100, 10)
    train_cut = np.percentile(train, cuts)
    train_cut = np.append(np.array([float('-Inf')]), train_cut, axis=0)
    train_cut = np.append(train_cut, np.array([float('Inf')]), axis=0)
    train_bins = pd.cut(train, train_cut).value_counts()
    oot_bins = pd.cut(oot, train_cut).value_counts()
    return cal_psi(train_bins, oot_bins)


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
        _x_train, _x_test = x.iloc[train_index, :], x.iloc[test_index, :]
        _y_train, _y_test = y[train_index], y[test_index]
        if weight is not None:
            clf.fit(_x_train, _y_train, sample_weight=weight[train_index])
        else:
            clf.fit(_x_train, _y_train, **kw)
        oos_pred = clf.predict_proba(_x_test)[:, 1]
        oot_pred = clf.predict_proba(x_oot)[:, 1]
        oos_idx.extend(test_index)
        stacking_train.extend(oos_pred)
        stacking_oot.append(oot_pred)
        auc_train.append(roc_auc_score(_y_train, clf.predict_proba(_x_train)[:, 1]))
        auc_test.append(roc_auc_score(_y_test, clf.predict_proba(_x_test)[:, 1]))
        auc_oot.append(roc_auc_score(y_oot, clf.predict_proba(x_oot)[:, 1]))
        ks_train.append(compute_ks(clf.predict_proba(_x_train)[:, 1], _y_train))
        ks_test.append(compute_ks(clf.predict_proba(_x_test)[:, 1], _y_test))
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
    if hasattr(opt, 'booster'):
        if opt.booster == 'dart':
            imp = opt.get_booster().get_score(importance_type='weight')
            feature_coef = pd.DataFrame(imp, index=['feature_coef']).T.reset_index()
            feature_coef = feature_coef.rename(columns={'index':'feature_name'})
    else:
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
    return feature_coef[['feature_name', 'feature_coef']]


def cal_iv(bad_vec, good_vec):
    _WOE_MIN = -20
    _WOE_MAX = 20
    woe = np.log((bad_vec / (good_vec + 0.001)) / (sum(bad_vec) / (sum(good_vec) + 0.001)))
    woe = woe.replace(-np.inf, _WOE_MIN)
    woe = woe.replace(np.inf, _WOE_MAX)
    iv = (bad_vec / sum(bad_vec) - good_vec / (sum(good_vec) + 0.001)) * woe
    return woe, iv


def feature_value_info(data_set, label_name, bin_split=10, oot_dm=None):
    data_matrix = {}
    if oot_dm:
        for x in tqdm(oot_dm):
            if x != label_name:
                if data_set[x].dtype == object:
                    _group = data_set.groupby(x)[label_name].agg(
                        {'bad_cnt': np.count_nonzero, 'obs_cnt': np.size})
                else:
                    dm = oot_dm[x].copy()
                    if isinstance(dm['value'][0], pd._libs.interval.Interval):
                        x_cut = pd.cut(data_set[x], bins=pd.IntervalIndex.from_intervals(dm['value']))
                        _group = data_set.groupby(x_cut)[label_name].agg(
                            {'bad_cnt': np.count_nonzero, 'obs_cnt': np.size})
                    else:
                        _group = data_set.groupby(x)[label_name].agg(
                            {'bad_cnt': np.count_nonzero, 'obs_cnt': np.size})
                _group.index.name = 'value'
                _group = _group.reset_index()
                _group["good_cnt"] = _group["obs_cnt"] - _group["bad_cnt"]
                _group["good_rate"] = _group["good_cnt"] / (_group["obs_cnt"])
                _group["bad_rate"] = _group["bad_cnt"] / (_group["obs_cnt"])
                _group['woe'], _group['iv'] = cal_iv(_group["bad_cnt"], _group["good_cnt"])
                _group.loc[_group['bad_cnt']<10, 'iv'] = 0.0
                data_matrix[x] = _group
    else:
       for x in tqdm(data_set.columns):
            if x != label_name:
                if data_set[x].dtype == object:
                    _group = data_set.groupby(x)[label_name].agg(
                        {'bad_cnt': np.count_nonzero, 'obs_cnt': np.size})
                else:
                    if data_set[x].nunique() <= bin_split:
                        _group = data_set.groupby(x)[label_name].agg(
                            {'bad_cnt': np.count_nonzero, 'obs_cnt': np.size})
                    else:
                        x_cut = pd.qcut(data_set[x], q=10, duplicates='drop')
                        _group = data_set.groupby(x_cut)[label_name].agg(
                            {'bad_cnt': np.count_nonzero, 'obs_cnt': np.size})
                if len(_group) <= 1:
                    continue
                _group.index.name = 'value'
                _group = _group.reset_index()
                _group["good_cnt"] = _group["obs_cnt"] - _group["bad_cnt"]
                _group["good_rate"] = _group["good_cnt"] / (_group["obs_cnt"])
                _group["bad_rate"] = _group["bad_cnt"] / (_group["obs_cnt"])
                _group['woe'], _group['iv'] = cal_iv(_group["bad_cnt"], _group["good_cnt"])
                _group.loc[_group['bad_cnt']<10, 'iv'] = 0.0
                data_matrix[x] = _group
    return data_matrix


def cal_psi(actual_cnts, expect_cnts):
    actual = actual_cnts / sum(actual_cnts)
    expect = expect_cnts / sum(expect_cnts)
    actual = actual.replace(0, 0.001)
    expect = expect.replace(0, 0.001)
    return np.sum((actual - expect) * np.log(actual / expect))


def tag_psi(data_matrix, data_matrix_oot, tag='obs_cnt'):
    psi = {}
    for col in data_matrix_oot:
        _df = pd.merge(data_matrix[col][['value', tag]],
                       data_matrix_oot[col][['value', tag]].rename(columns={tag: tag + '_oot'}),
                       how='inner',
                       on='value')
        psi[col] = cal_psi(_df[tag], _df[tag + '_oot'])
    return psi


def get_describe(df):
    """
    数据描述， 空值和最常值
    :param x:
    :return:
    """
    nu = []
    nu_ratio = []
    most_common = []
    most_common_ratio = []
    for i in tqdm(df.columns):
        if len(df[i].value_counts()) == 0:
            most_common.append(len(df[i]))
            most_common_ratio.append(1.0)
        else:
            most_common.append(sum(df[i] == (df[i].value_counts().index[0])))
            most_common_ratio.append(sum(df[i] == (df[i].value_counts().index[0])) / df.shape[0])
        nu.append(sum(df[i].isnull()))
        nu_ratio.append(sum(df[i].isnull()) / df.shape[0])

    # 变量初筛
    sheet_2_tmp = pd.merge(
        pd.DataFrame({'变量名称': list(df.columns),
                      '空值个数': nu,
                      '空值个数占比': nu_ratio,
                      '最常值个数': most_common,
                      '最常值个数占比': most_common_ratio}),
        pd.DataFrame(
            df.describe().T.reset_index()).rename(
            columns={'index': '变量名称'}),
        how='left', on='变量名称')
    return sheet_2_tmp


def woe_to_sql(woe_dict):

    seq = 'when {score_name} >= {lower} and {score_name} < {upper} then {value} \n'
    seq_nan = 'when {score_name} is null then {value} \n'
    res_lst = []
    for f in woe_dict:
        f_lst = []
        for threshold in woe_dict[f]:
            woe_value = woe_dict[f][threshold]
            threshold = threshold.replace('[', '').replace(' ', '').replace(')', '')
            left, right = threshold.split(',')
            if left == 'nan':
                f_lst.append(seq_nan.format(score_name=f, value=woe_value))
            else:
                f_lst.append(seq.format(score_name=f, lower=left, upper=right, value=woe_value))
        f_str = 'case ' + ''.join(f_lst) + 'end as {woe_name}, \n'.format(woe_name=f+'_woe')
        res_lst.append(f_str)
    return ''.join(res_lst)

