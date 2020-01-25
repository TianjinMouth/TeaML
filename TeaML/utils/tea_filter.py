import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMClassifier
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from .tea_utils import get_importance, feature_select
import shap
import numpy as np


class FilterVif(TransformerMixin, BaseEstimator):
    def __init__(self, vif_threshold=15):
        self.vif_threshold = vif_threshold
        self.left_features = []
        self.vif_result = None

    def fit(self, x):
        x_train_matrix = x.values
        vif_list = [variance_inflation_factor(x_train_matrix, i) for i in range(x_train_matrix.shape[1])]
        vif_result = pd.DataFrame({'字段名称': x.columns.tolist(), 'VIF': vif_list})
        tmp = vif_result[vif_result['VIF'] < self.vif_threshold]
        self.left_features = tmp['字段名称'].tolist()
        self.vif_result = vif_result
        return self

    def transform(self, x):
        return x[self.left_features]


class FilterModel(TransformerMixin, BaseEstimator):
    def __init__(self, model_type='lgb', left_features_num=35):
        self.left_features_num = left_features_num
        self.model_type = model_type
        self.left_features = []

    def fit(self, x, y):
        if self.model_type == 'lgb':
            clf = LGBMClassifier(max_depth=-1, learning_rate=0.1, n_estimators=30, class_weight='balanced',
                                 reg_alpha=11, num_leaves=9, boosting_type='gbdt')
        elif self.model_type == 'lr':
            clf = LogisticRegression(penalty='l1', C=0.1)
        elif self.model_type == 'rf':
            clf = RandomForestClassifier(n_estimators=50, criterion='gini', min_weight_fraction_leaf=0.0,
                                         max_leaf_nodes=None, bootstrap=True, oob_score=False, class_weight='balanced')
        else:
            raise KeyError("Wrong type...")
        clf.fit(x, y)
        im = get_importance(clf, x)
        self.left_features = im['feature_name'][:self.left_features_num]
        return self

    def transform(self, x):
        return x[self.left_features]


class FilterIV(TransformerMixin, BaseEstimator):
    def __init__(self, fst_keep=80, left_features_num=35):
        self.fst_keep = fst_keep
        self.left_features_num = left_features_num
        self.left_features = []
        self.mm = MinMaxScaler()
        self.col = []

    def fit(self, x, y):
        self.col = x.columns.tolist()
        self.mm.fit(x, y)
        x_minmax = self.mm.transform(x)
        x_minmax = pd.DataFrame(x_minmax, columns=self.col)
        self.left_features = feature_select(x_minmax, y, method='iv', kb=self.fst_keep, rfe=self.left_features_num)
        return self

    def transform(self, x):
        return x[self.left_features]


class FilterSHAP(TransformerMixin, BaseEstimator):
    def __init__(self, left_features_num=None):
        self.left_features_num = left_features_num
        self.left_features = []

    def fit(self, x, y):
        clf = LGBMClassifier(max_depth=-1, learning_rate=0.1, n_estimators=30, class_weight='balanced',
                             reg_alpha=11, num_leaves=9, boosting_type='gbdt')
        clf.fit(x, y)
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(x)
        shape_importance = pd.concat([pd.DataFrame(x.columns, columns=['feature_name']),
                                      pd.DataFrame(np.sum(np.abs(shap_values), axis=0), columns=['importance'])],
                                     axis=1)
        shape_importance['idx'] = shape_importance.index
        shape_importance = shape_importance.sort_values(['importance'], ascending=False).reset_index(drop=True)
        if self.left_features_num:
            self.left_features = shape_importance.loc[:self.left_features_num, :]['feature_name'].tolist()
        else:
            self.left_features = shape_importance[shape_importance['importance'] > 0]['feature_name'].tolist()
        return self

    def transform(self, x):
        return x[self.left_features]


class FilterANOVA(TransformerMixin, BaseEstimator):
    def __init__(self, fst_keep=80, left_features_num=35):
        self.fst_keep = fst_keep
        self.left_features_num = left_features_num
        self.left_features = []
        self.mm = MinMaxScaler()
        self.col = []

    def fit(self, x, y):
        self.col = x.columns.tolist()
        self.mm.fit(x, y)
        x_minmax = self.mm.transform(x)
        x_minmax = pd.DataFrame(x_minmax, columns=self.col)
        self.left_features = feature_select(x_minmax, y, method='f', kb=self.fst_keep, rfe=self.left_features_num)
        return self

    def transform(self, x):
        return x[self.left_features]


class FilterCoLine(TransformerMixin, BaseEstimator):
    def __init__(self, params):
        self.left_features = []
        self.params = params

    def fit(self, x, y):
        _X = x.copy()
        while True:
            # lr = Ridge(alpha=0.1, fit_intercept=False)
            # lr.fit(_X, y)
            lr = LogisticRegression(**self.params)
            lr.fit(_X, y)
            co_line = pd.DataFrame({'features': _X.columns, 'coe': lr.coef_[0]}).sort_values('coe')
            self.left_features = co_line[co_line['coe'] > 0]['features'].tolist()
            if (co_line['coe'] < 0).sum() == 0:
                break
            _X = _X[self.left_features]
            print("LR 筛除权重为负的特征， 剩余%s" % _X.shape[1])
        return self

    def transform(self, x):
        return x[self.left_features]


class OutlierTransform(TransformerMixin, BaseEstimator):
    def __init__(self, limit_value=20, method='box', percentile_limit_set=90, changed_feature_box=[]):
        """

        :param limit_value: 最小处理样本个数set, 当独立样本大于limit_value, 可以CategoricalTransform或One_hot
        :param method: 'box' or 'self_def', default 'box'
        :param percentile_limit_set: default 90
        :param changed_feature_box: default []
        """
        self.limit_value = limit_value
        self.method = method
        self.percentile_limit_set = percentile_limit_set
        self.changed_feature_box = changed_feature_box
        self.box_result = dict()

    def fit(self, x):
        data = x.copy()
        if self.method == 'box':
            for i in data.columns:
                if len(pd.DataFrame(data[i]).drop_duplicates()) >= self.limit_value:
                    q1 = np.percentile(np.array(data[i][~data[i].isnull()]), 25)
                    q3 = np.percentile(np.array(data[i][~data[i].isnull()]), 75)
                    top = q3 + 1.5 * (q3 - q1)
                    self.box_result[i] = top

        if self.method == 'self_def':
            # 快速截断
            if len(self.changed_feature_box) == 0:
                # 当方法选择为自定义，且没有定义changed_feature_box则全量数据全部按照percentile_limit_set的分位点大小进行截断
                for i in data.columns:
                    if len(pd.DataFrame(data[i]).drop_duplicates()) >= self.limit_value:
                        q_limit = np.percentile(np.array(data[i][~data[i].isnull()]),
                                                self.percentile_limit_set)
                        self.box_result[i] = q_limit
            else:
                # 如果定义了changed_feature_box，则将changed_feature_box里面的按照box方法，其余快速截断
                for i in data.columns:
                    if len(pd.DataFrame(data[i]).drop_duplicates()) >= self.limit_value:
                        if i in self.changed_feature_box:
                            q1 = np.percentile(np.array(data[i][~data[i].isnull()]), 25)
                            q3 = np.percentile(np.array(data[i][~data[i].isnull()]), 75)
                            top = q3 + 1.5 * (q3 - q1)
                            self.box_result[i] = top
                        else:
                            q_limit = np.percentile(np.array(data[i][~data[i].isnull()]),
                                                    self.percentile_limit_set)
                            self.box_result[i] = q_limit

        return self

    def transform(self, x):
        test = x.copy()
        for i in self.box_result:
            if self.box_result[i] == 0:
                pass
            else:
                test[i][test[i] > self.box_result[i]] = self.box_result[i]
        return test


class FilterCorr(TransformerMixin, BaseEstimator):
    def __init__(self, k=35):
        self.k = k
        self.new_c = []

    def fit(self, x_train):
        _x = x_train[[x for x in x_train.columns]]
        res = np.abs(np.corrcoef(_x.T))
        vif_value = []
        for i in range(res.shape[0]):
            for j in range(res.shape[0]):
                if j > i:
                    vif_value.append([_x.columns[i], _x.columns[j], res[i, j]])
        vif_value = sorted(vif_value, key=lambda x: x[2])
        if self.k is not None:
            if self.k < len(vif_value):
                for i in range(len(_x)):
                    if vif_value[-i][1] not in self.new_c:
                        self.new_c.append(vif_value[-i][1])
                    else:
                        self.new_c.append(vif_value[-i][0])
                    if len(self.new_c) == self.k:
                        break
                return self
            else:
                print('feature个数越界')
        else:
            return self

    def transform(self, x_oot):
        return x_oot[self.new_c]


class FilterStepWise(TransformerMixin, BaseEstimator):
    def __init__(self, left_features_num=None, method='p_value', threshold_in=0.01, threshold_out=0.05, verbose=True):
        self.left_features_num = left_features_num
        self.left_features = []
        self.initial_list = []
        self.method = method
        self.threshold_in = threshold_in
        self.threshold_out = threshold_out
        self.verbose = verbose

    def fit(self, x, y):
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        if self.method == 'p_value':
            """
            Perform a forward-backward feature selection
            based on p-value from statsmodels.api.OLS
            Arguments:
                X - pandas.DataFrame with candidate features
                y - list-like with the target
                initial_list - list of features to start with (column names of X)
                threshold_in - include a feature if its p-value < threshold_in
                threshold_out - exclude a feature if its p-value > threshold_out
                verbose - whether to print the sequence of inclusions and exclusions
            Returns: list of selected features
            Always set threshold_in < threshold_out to avoid infinite looping.
            See https://en.wikipedia.org/wiki/Stepwise_regression for the details
            """
            included = list(self.initial_list)

            while True:
                changed = False
                # forward step
                excluded = list(set(x.columns) - set(included))
                new_pval = pd.Series(index=excluded)
                for new_column in excluded:
                    model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included + [new_column]]))).fit()
                    new_pval[new_column] = model.pvalues[new_column]
                best_pval = new_pval.min()
                if best_pval < self.threshold_in:
                    best_feature = new_pval.argmin()
                    included.append(best_feature)
                    changed = True
                    if self.verbose:
                        print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

                # backward step
                model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included]))).fit()
                # use all coefs except intercept
                pvalues = model.pvalues.iloc[1:]
                worst_pval = pvalues.max()  # null if pvalues is empty
                if worst_pval > self.threshold_out:
                    changed = True
                    worst_feature = pvalues.argmax()
                    included.remove(worst_feature)
                    if self.verbose:
                        print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
                if not changed:
                    break
                self.left_features = included.copy()[:self.left_features_num]
        elif self.method == 'r_squared':
            """
            前向逐步回归算法，来自https://planspace.org/20150423-forward_selection_with_statsmodels/
            使用Adjusted R-squared来评判新加的参数是否提高回归中的统计显著性
            Linear model designed by forward selection.
            Parameters:
            -----------
            data : pandas DataFrame with all possible predictors and response
            response: string, name of response column in data
            Returns:
            --------
            model: an "optimal" fitted statsmodels linear model
                   with an intercept
                   selected by forward selection
                   evaluated by adjusted R-squared
            """
            remaining = set(x.columns)
            response = y.name
            selected = self.initial_list
            current_score, best_new_score = 0.0, 0.0
            step = self.left_features_num
            while remaining and current_score == best_new_score and step > 0:
                scores_with_candidates = []
                for candidate in remaining:
                    formula = "{} ~ {} + 1".format(response, ' + '.join(selected + [candidate]))
                    score = smf.ols(formula, pd.concat([x, y], axis=1)).fit().rsquared_adj
                    scores_with_candidates.append((score, candidate))
                scores_with_candidates.sort()
                best_new_score, best_candidate = scores_with_candidates.pop()
                if current_score < best_new_score:
                    remaining.remove(best_candidate)
                    selected.append(best_candidate)
                    current_score = best_new_score
                    if self.verbose:
                        print('Current columns are %s' % selected)
                        print('Current R^2 is %f' % current_score)
                step -= 1

            self.left_features = selected.copy()
        elif self.method in ('AIC', 'BIC'):
            selected = set(x.columns)
            response = y.name
            remaining = self.initial_list
            if self.method == 'AIC':
                best_new_score, current_score = 0.0, smf.ols("{} ~ {} + 1".format(response, ' + '.join(selected)),
                                                             pd.concat([x, y], axis=1)).fit().aic
                print('Start AIC/BIC is %f' % current_score)
            else:
                best_new_score, current_score = 0.0, smf.ols("{} ~ {} + 1".format(response, ' + '.join(selected)),
                                                             pd.concat([x, y], axis=1)).fit().bic
                print('Start AIC/BIC is %f' % current_score)
            step = self.left_features_num
            while selected and step > 0:
                scores_with_candidates = []
                for candidate in selected:
                    formula = "{} ~ {} + 1".format(response, ' + '.join(selected - set([candidate])))
                    if self.method == 'AIC':
                        score = smf.ols(formula, pd.concat([x, y], axis=1)).fit().aic
                    else:
                        score = smf.ols(formula, pd.concat([x, y], axis=1)).fit().bic
                        print('score: %s' % score)
                    scores_with_candidates.append((score, candidate))
                scores_with_candidates.sort()
                best_new_score, best_candidate = scores_with_candidates.pop(0)
                print(best_new_score, best_candidate)
                if current_score > best_new_score:
                    remaining.append(best_candidate)
                    selected.remove(best_candidate)
                    current_score = best_new_score
                    if self.verbose:
                        print('Current columns are %s' % selected)
                        print('Current AIC/BIC is %f' % current_score)
                    step -= 1
                else:
                    break

            self.left_features = selected.copy()
        else:
            print('请输入p_value/r_squared/AIC/BIC')

        return self

    def transform(self, x):
        return x[self.left_features]
