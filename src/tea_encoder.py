import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from itertools import product
from sklearn.base import TransformerMixin, BaseEstimator, clone


class TeaOneHotEncoder(TransformerMixin, BaseEstimator):
    """
    one_hot_encoder
    """
    def __init__(self, num=10):
        self.dictionary = dict()
        self.num = num
        self.continuous_col = []
        self.categorical_col = []
        self.oh_dict = dict()

    def check_types(self, x, replace=True):
        if replace:
            self.continuous_col = []
            self.categorical_col = []
        for f in x.columns:
            x_ = x[f]
            try:
                x_ = x_.astype('float')
                counts = len(x_.drop_duplicates())
                if counts <= self.num:
                    self.categorical_col.append(f)
                else:
                    self.continuous_col.append(f)
            except Exception:
                self.categorical_col.append(f)

    def fit(self, x):
        oh = OneHotEncoder(handle_unknown='ignore', categories='auto')
        self.check_types(x)
        if len(self.categorical_col) > 0:
            for f in self.categorical_col:
                x[f] = x[f].astype(str)
                _one_hot = clone(oh)
                _one_hot.fit(x[[f]])
                self.oh_dict[f] = _one_hot
            return self
        else:
            print('Please check out your categorical variance')

    def transform(self, x):
        for c in self.categorical_col:
            x[c] = x[c].astype(str)
            _tmp = self.oh_dict[c].transform(x[[c]]).toarray()
            _tmp_df = pd.DataFrame(_tmp, columns=[c + '_' + str(i) for i in self.oh_dict[c].categories_[0]])
            x = pd.concat([x, _tmp_df], axis=1)
        return x.drop(self.categorical_col, axis=1)


class TeaBadRateEncoder(TransformerMixin, BaseEstimator):
    """
    bad_rate替换
    """
    def __init__(self, num=10):
        self.categorical_var = []
        self.dictionary = dict()
        self.num = num
        self.continuous_col = []
        self.categorical_col = []

    def check_types(self, x, replace=True):
        if replace:
            self.continuous_col = []
            self.categorical_col = []
        for f in x.columns:
            x_ = x[f]
            try:
                x_ = x_.astype('float')
                counts = len(x_.drop_duplicates())
                if counts <= self.num:
                    self.categorical_col.append(f)
                else:
                    self.continuous_col.append(f)
            except Exception:
                self.categorical_col.append(f)

    def fit(self, x, y):
        """
        分类变量进行连续化变换 bad_rate替换
        :param x: pd.DataFrame categorical
        :param y: pd.Series label
        :return: CategoricalTransform对象
        """

        self.check_types(x)
        if len(self.categorical_col) > 0:
            data_cate = x.loc[:, self.categorical_col]
            nan_rate = data_cate.apply(self._nan_transform, axis=0)
            data_cate["label"] = y
            for elem in self.categorical_col:
                temp = data_cate.groupby(elem)["label"].mean().sort_values(ascending=False)
                temp.set_value(np.nan, value=nan_rate[elem])
                self.dictionary.update({elem: temp})  # order
            return self
        else:
            print('Please fill in your categorical variance')

    @staticmethod
    def _nan_transform(x):
        return float(sum([str(elem) == "nan" for elem in x])) / len(x)

    def transform(self, x):
        """
        对离散变量连续化做transform变换
        :param x: pd.DataFrame 待transform离散变量数据集
        :return: pd.DataFrame 连续化transform结果
        """
        keys = self.dictionary.keys()
        for elem in x.columns:
            if elem not in keys:
                continue
            x[elem] = x[elem].apply(self._element_transform, col=elem)
        return x

    def _element_transform(self, element, col):
        if element in self.dictionary[col].index:
            return self.dictionary[col].loc[element]
        elif isinstance(element, str):
            return 0.0


class TeaMeanEncoder:
    def __init__(self, categorical_features, n_splits=5, target_type='classification', prior_weight_func=None):
        """
        :param categorical_features: list, the name of the categorical columns to encode
        :param n_splits: the number of splits used in mean encoding
        :param target_type: str, 'regression' or 'classification'
        :param prior_weight_func:
        a function that takes in the number of observations, and outputs prior weight
        when a dict is passed, the default exponential decay function will be used:
        k: the number of observations needed for the posterior to be weighted equally as the prior
        f: larger f --> smaller slope
        """

        self.categorical_features = categorical_features
        self.n_splits = n_splits
        self.learned_stats = {}

        if target_type == 'classification':
            self.target_type = target_type
            self.target_values = []
        else:
            self.target_type = 'regression'
            self.target_values = None

        if isinstance(prior_weight_func, dict):
            self.prior_weight_func = eval('lambda x: 1 / (1 + np.exp((x - k) / f))', dict(prior_weight_func, np=np))
        elif callable(prior_weight_func):
            self.prior_weight_func = prior_weight_func
        else:
            self.prior_weight_func = lambda x: 1 / (1 + np.exp((x - 10) / 1))

    @staticmethod
    def mean_encode_subroutine(x_train, y_train, x_test, variable, target, prior_weight_func):
        x_train = x_train[[variable]].copy()
        x_test = x_test[[variable]].copy()

        if target is not None:
            nf_name = '{}_pred_{}'.format(variable, target)
            x_train['pred_temp'] = (y_train == target).astype(int)  # classification
        else:
            nf_name = '{}_pred'.format(variable)
            x_train['pred_temp'] = y_train  # regression
        prior = x_train['pred_temp'].mean()

        col_avg_y = x_train.groupby(by=variable, axis=0)['pred_temp'].agg({'mean': 'mean', 'beta': 'size'})
        col_avg_y['beta'] = prior_weight_func(col_avg_y['beta'])
        col_avg_y[nf_name] = col_avg_y['beta'] * prior + (1 - col_avg_y['beta']) * col_avg_y['mean']
        col_avg_y.drop(['beta', 'mean'], axis=1, inplace=True)

        nf_train = x_train.join(col_avg_y, on=variable)[nf_name].values
        nf_test = x_test.join(col_avg_y, on=variable).fillna(prior, inplace=False)[nf_name].values

        return nf_train, nf_test, prior, col_avg_y

    def fit_transform(self, x, y):
        """
        :param x: pandas DataFrame, n_samples * n_features
        :param y: pandas Series or numpy array, n_samples
        :return x_new: the transformed pandas DataFrame containing mean-encoded categorical features
        """
        x_new = x.copy()
        if self.target_type == 'classification':
            skf = StratifiedKFold(self.n_splits)
        else:
            skf = KFold(self.n_splits)

        if self.target_type == 'classification':
            self.target_values = sorted(set(y))
            self.learned_stats = {'{}_pred_{}'.format(variable, target): [] for variable, target in
                                  product(self.categorical_features, self.target_values)}
            for variable, target in product(self.categorical_features, self.target_values):
                nf_name = '{}_pred_{}'.format(variable, target)
                x_new.loc[:, nf_name] = np.nan
                for large_ind, small_ind in skf.split(y, y):
                    nf_large, nf_small, prior, col_avg_y = TeaMeanEncoder.mean_encode_subroutine(
                        x_new.iloc[large_ind], y.iloc[large_ind], x_new.iloc[small_ind], variable, target,
                        self.prior_weight_func)
                    x_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))
        else:
            self.learned_stats = {'{}_pred'.format(variable): [] for variable in self.categorical_features}
            for variable in self.categorical_features:
                nf_name = '{}_pred'.format(variable)
                x_new.loc[:, nf_name] = np.nan
                for large_ind, small_ind in skf.split(y, y):
                    nf_large, nf_small, prior, col_avg_y = TeaMeanEncoder.mean_encode_subroutine(
                        x_new.iloc[large_ind], y.iloc[large_ind], x_new.iloc[small_ind], variable, None,
                        self.prior_weight_func)
                    x_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))

        x_new = x_new.drop(['{}_pred_0'.format(variable) for variable in self.categorical_features], axis=1)
        x_new = x_new.drop([variable for variable in self.categorical_features], axis=1)
        x_new = x_new.rename(columns={'{}_pred_1'.format(variable): variable for variable in self.categorical_features})
        return x_new

    def transform(self, x):
        """
        :param x: pandas DataFrame, n_samples * n_features
        :return x_new: the transformed pandas DataFrame containing mean-encoded categorical features
        """
        x_new = x.copy()

        if self.target_type == 'classification':
            for variable, target in product(self.categorical_features, self.target_values):
                nf_name = '{}_pred_{}'.format(variable, target)
                x_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    x_new[nf_name] += x_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
                        nf_name]
                x_new[nf_name] /= self.n_splits
        else:
            for variable in self.categorical_features:
                nf_name = '{}_pred'.format(variable)
                x_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    x_new[nf_name] += x_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
                        nf_name]
                x_new[nf_name] /= self.n_splits

        x_new = x_new.drop(['{}_pred_0'.format(variable) for variable in self.categorical_features], axis=1)
        x_new = x_new.drop([variable for variable in self.categorical_features], axis=1)
        x_new = x_new.rename(columns={'{}_pred_1'.format(variable): variable for variable in self.categorical_features})

        return x_new


