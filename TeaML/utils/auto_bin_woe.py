from scipy import stats
import numpy as np
import pandas as pd
from tqdm import tqdm


class AutoBinWOE(object):
    """
    连续变量合并，使得woe值单调

    调用函数之前要进行以下处理：
    1. 异常值：
    - 异常值极少时（1%以下），将样本舍弃
    - 异常值较少时（20%以下），将异常值替换为空值
    - 异常值较多时（20%以上），考虑舍弃变量
    2. 空值：
    - 缺省值很多时（40%以上）直接舍弃。作为特征加入的话，可能反倒带入噪声，影响最后的结果。
    - 非连续特征缺省量适中时（10%-40%），将Nan作为一个新类别加入至特征中
    - 连续特征缺省量适中时（10%-40%），考虑给定一个step(比如age，我们可以考虑每隔2/3岁为一个步长)，然后把它离散化，之后把NaN作为一个type加到
        属性类目中
    - 缺省值很少时（10%以下）利用填充的办法进行处理。例如用均值、中位数、众数填充，模型填充等

    函数结果关注：
    1. IV值为+∞处理
    - 如果可能，直接把这个分组做成一个规则，作为模型的前置条件或补充条件；
    - 重新对变量进行离散化或分组，使每个分组的响应比例都不为0且不为100%，尤其是当一个分组个体数很小时（比如小于100个），强烈建议这样做，因为本身把
        一个分组个体数弄得很小就不是太合理。
    - 如果上面两种方法都无法使用，建议人工把该分组的响应数和非响应的数量进行一定的调整。如果响应数原本为0，可以人工调整响应数为1，如果非响应数原本
        为0，可以人工调整非响应数为1.

    2. IV值的判断
    若IV信息量取值小于0.02，认为该指标对因变量没有预测能力，应该被剔除；
    若IV信息量取值在0.02与0.1之间，认为该指标对因变量有较弱的预测能力；
    若IV信息量取值在0.1与0.3之间，认为该指标对因变量的预测能力一般；
    若IV信息量取值大于0.3，认为该指标对因变量有较强的预测能力。
    实际应用时，可以保留IV值大于0.1的指标。
    """
    def __init__(self, bins=10, num=1, monotony_merge=True, bad_rate_merge=False, bad_rate_sim_threshold=0.05,
                 chi2_merge=False, chi2_threshold=2.706, prune=False, prune_threshold=0.05):
        """

        :param bins: 初始分箱个数
        :param num: 变量类别数，区分分类和连续变量
        :param bad_rate_merge: 相近的bad_rate是否要合并
        :param bad_rate_sim_threshold: 相似性阈值，mean_bad_rate的乘数
        :param chi2_merge: 相近的X^2是否要合并
        :param chi2_threshold: X^2阈值
        """
        self.threshold = None
        self.bins = bins
        self.num = num
        self.data_matrix = dict()
        self.continuous_col = []
        self.categorical_col = []
        self._WOE_MIN = -20
        self._WOE_MAX = 20
        self.bad_rate_maps = {}
        self.type_checked = 0
        self.data_matrix_origin = dict()
        self.is_bad_rate_replace = 0
        self.mean_bad_rate = 0
        self.monotony_merge = monotony_merge
        self.bad_rate_merge = bad_rate_merge
        self.bad_rate_sim_threshold = bad_rate_sim_threshold
        self.chi2_merge = chi2_merge
        self.chi2_threshold = chi2_threshold
        self.prune = prune
        self.prune_threshold = prune_threshold

    def check_types(self, x, replace=True):
        """
        区分连续变量和离散变量
        :param x:
        :param replace: 将保存的变量列表清空
        :return:
        """
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
        self.type_checked = 1

    def _bin_fit(self, x):
        """
        分bin的fit，将阈值保存
        :param x:
        :return:
        """
        self.check_types(x, replace=True)
        thresholds = {}
        for col in self.continuous_col:
            x_ = x[col].values
            thr = self._bin_single_fit(x_)
            thresholds[col] = thr

        self.threshold = thresholds

    def _bin_single_fit(self, x):
        """
        bin the input 1-D numpy array using n equal percentiles
        :param x: 1-D numpy array
        :return: threshold dict
        """
        threshold = {}
        res = x[~np.isnan(x)].copy()
        for i in range(self.bins):
            point1 = stats.scoreatpercentile(res, i * (100 / self.bins))
            point2 = stats.scoreatpercentile(res, (i + 1) * (100 / self.bins))
            if i == self.bins-1:
                point2 = point2 + 1e-1
            threshold[i] = [point1, point2]
        return threshold

    def _bin_transform(self, x):
        """
        将阈值transform到新的样本上
        :param x:
        :return:
        """
        tmp = []
        for col in self.continuous_col:
            x_ = x[col].values
            thr = self.threshold[col]
            res = x[col].copy()
            for k in thr.keys():
                point1, point2 = thr[k]
                x1 = x_[np.where((x_ >= point1) & (x_ < point2))]
                mask = np.in1d(x_, x1)
                res[mask] = k
            res[np.isnan(x_)] = -1
            tmp.append(res)
        return pd.DataFrame(np.array(tmp).T, columns=self.continuous_col)

    def _bin_fit_transform(self, x):
        self._bin_fit(x)
        return self._bin_transform(x)

    def _category_feature(self, x, y):
        data = pd.DataFrame({"var": x, "label": list(y)})
        data_matrix = data.groupby("var", as_index=False)["label"].agg(
            {'bad': np.count_nonzero, 'obs': np.size})
        data_matrix["good"] = data_matrix["obs"] - data_matrix["bad"]
        data_matrix["good_rate"] = data_matrix["good"] / data_matrix["obs"]
        data_matrix["bad_rate"] = data_matrix["bad"] / data_matrix["obs"]
        data_matrix["lower"] = data_matrix.index.tolist()
        return data_matrix

    def monotony_single_fit(self, x, y, threshold):
        """
        对cut labels 进行合并迭代,寻找使得abs(Spearman coefficient)最大的labels组合
        :param x: 原始数据列向量, pd.Series or np.array
        :param y: list or np.array or pd.Series
        :param threshold:
        :return: new cut label and the best Spearman coefficient
        """
        data = pd.DataFrame({"var": x, "label": list(y)})
        data_group = data.groupby("var", as_index=False)["label"].agg(
            {'bad': np.count_nonzero, 'obs': np.size})

        # 保留一份单调性合并前的分bin矩阵，用作数据分析
        data_group_origin = data_group.copy()
        threshold_df = pd.DataFrame([[i[0], i[1]] for i in threshold.values()], index=threshold.keys(),
                                    columns=['left', 'right'])
        data_group_origin = data_group_origin.merge(threshold_df, how='left', left_on='var', right_index=True)
        data_group_origin.index = np.array(["bin_" + str(i) for i in range(data_group_origin.shape[0])])
        data_group_origin["good"] = data_group_origin["obs"] - data_group_origin["bad"]
        data_group_origin["good_rate"] = data_group_origin["good"] / data_group_origin["obs"]
        data_group_origin["bad_rate"] = data_group_origin["bad"] / data_group_origin["obs"]

        # 将空值单调取出，不参与合并
        nan_bin = data_group[data_group['var'] == -1]
        data_group = data_group[data_group['var'] != -1].reset_index(drop=True)

        seq = data_group['var'].values
        bad = data_group['bad'].values
        obs = data_group['obs'].values
        coef, _ = stats.spearmanr((bad / obs), seq)
        best_bins = {"coef": abs(coef), "seq": seq, "bad": bad, "obs": obs}
        if self.monotony_merge:
            # 单调性合并
            while best_bins['coef'] < 0.9999:
                rlst = list()
                for i in range(len(bad) - 1):
                    bad_temp = np.delete(bad, i)
                    bad_temp[i] += bad[i]
                    obs_temp = np.delete(obs, i)
                    obs_temp[i] += obs[i]
                    seq_temp = np.delete(seq, i)
                    r, p = stats.spearmanr((bad_temp / obs_temp), seq_temp)
                    rlst.append(abs(r))
                    if abs(r) >= best_bins['coef']:
                        best_bins['coef'] = abs(r)
                        best_bins['seq'] = seq_temp
                        best_bins['bad'] = bad_temp
                        best_bins['obs'] = obs_temp
                # 处理当减少一个bin单调性反而下降，导致陷入死循环的问题
                if (max(rlst) < best_bins['coef']) & (best_bins['coef'] < 0.9999):
                    best_bins['coef'] = 0.0
                seq = best_bins['seq']
                bad = best_bins['bad']
                obs = best_bins['obs']

        # 是否进行bad_rate相似性合并
        if self.bad_rate_merge:
            seq = best_bins['seq']
            bad = best_bins['bad']
            obs = best_bins['obs']
            best_bins = {"seq": seq, "bad": bad, "obs": obs}
            if len(seq) > 2:
                diff = [abs(best_bins['bad'][i] / best_bins['obs'][i] - best_bins['bad'][i + 1] /
                            best_bins['obs'][i + 1])
                        for i in range(len(best_bins['obs']) - 1)]
                while min(diff) <= (self.mean_bad_rate * self.bad_rate_sim_threshold):
                    for i in range(len(diff)):
                        if diff[i] <= (self.mean_bad_rate * self.bad_rate_sim_threshold):
                            best_bins['bad'] = np.delete(bad, i)
                            best_bins['bad'][i] += bad[i]
                            best_bins['obs'] = np.delete(obs, i)
                            best_bins['obs'][i] += obs[i]
                            best_bins['seq'] = np.delete(seq, i)
                    seq = best_bins['seq']
                    bad = best_bins['bad']
                    obs = best_bins['obs']
                    diff = [abs(best_bins['bad'][i] / best_bins['obs'][i] - best_bins['bad'][i + 1] /
                                best_bins['obs'][i + 1])
                            for i in range(len(best_bins['obs']) - 1)]
                    if len(diff) < 2:
                        break

        # 是否进行卡方合并
        if self.chi2_merge:
            seq = best_bins['seq']
            bad = best_bins['bad']
            obs = best_bins['obs']
            best_bins = {"seq": seq, "bad": bad, "obs": obs}
            if len(seq) > 2:
                chi2 = [(self.mean_bad_rate * best_bins['obs'][i] - best_bins['bad'][i]) ** 2 /
                        self.mean_bad_rate * best_bins['obs'][i]
                        for i in range(len(seq))]
                while max(chi2) >= self.chi2_threshold:
                    for i in range(len(chi2)-1):
                        if chi2[i] >= self.chi2_threshold:
                            best_bins['bad'] = np.delete(bad, i)
                            best_bins['bad'][i] += bad[i]
                            best_bins['obs'] = np.delete(obs, i)
                            best_bins['obs'][i] += obs[i]
                            best_bins['seq'] = np.delete(seq, i)
                    seq = best_bins['seq']
                    bad = best_bins['bad']
                    obs = best_bins['obs']
                    chi2 = [(self.mean_bad_rate * best_bins['obs'][i] - best_bins['bad'][i]) ** 2 / self.mean_bad_rate *
                            best_bins['obs'][i] for i in range(len(seq))]
                    if len(chi2) < 2:
                        break

        # 是否防止过拟合
        if self.prune:
            seq = best_bins['seq']
            bad = best_bins['bad']
            obs = best_bins['obs']
            best_bins = {"seq": seq, "bad": bad, "obs": obs}
            if len(seq) > 2:
                diff = [abs(best_bins['bad'][i] / best_bins['obs'][i] - best_bins['bad'][i + 1] /
                            best_bins['obs'][i + 1])
                        for i in range(len(best_bins['obs']) - 1)]
                while max(diff) >= self.prune_threshold:
                    for i in range(len(diff)):
                        if diff[i] >= self.prune_threshold:
                            best_bins['bad'] = np.delete(bad, i)
                            best_bins['bad'][i] += bad[i]
                            best_bins['obs'] = np.delete(obs, i)
                            best_bins['obs'][i] += obs[i]
                            best_bins['seq'] = np.delete(seq, i)
                    seq = best_bins['seq']
                    bad = best_bins['bad']
                    obs = best_bins['obs']
                    diff = [abs(best_bins['bad'][i] / best_bins['obs'][i] - best_bins['bad'][i + 1] /
                                best_bins['obs'][i + 1])
                            for i in range(len(best_bins['obs']) - 1)]
                    if len(diff) < 2:
                        break

        left = [threshold[0][0]]
        right = []
        for s in best_bins['seq'][:-1]:
            left.append(threshold[s + 1][0])
        for s in best_bins['seq']:
            right.append(threshold[s][1])

        data_group_best = pd.DataFrame({"bad": best_bins['bad'], "obs": best_bins['obs'], "left": left, "right": right})
        data_group_best = data_group_best[['bad', 'obs', 'left', 'right']]
        data_group_best = pd.concat([data_group_best, nan_bin.drop(['var'], axis=1)])
        data_group_best["good"] = data_group_best["obs"] - data_group_best["bad"]
        data_group_best["good_rate"] = data_group_best["good"] / data_group_best["obs"]
        data_group_best["bad_rate"] = data_group_best["bad"] / data_group_best["obs"]
        data_group_best = data_group_best.sort_values('bad_rate')
        data_group_best.index = np.array(["bin_" + str(i) for i in range(data_group_best.shape[0])])
        return data_group_best, data_group_origin

    def fit(self, x, y):
        x_bin = self._bin_fit_transform(x)
        self.mean_bad_rate = y.mean()
        for variable in tqdm(x_bin.columns):
            if variable in self.continuous_col:
                threshold = self.threshold[variable]
                data_matrix, data_group_origin = self.monotony_single_fit(x_bin[variable], y, threshold)
                self.data_matrix_origin.update({variable: data_group_origin})
            else:
                data_matrix = self._category_feature(x_bin[variable], y)
            self.calc_woe(data_matrix)
            self.data_matrix.update({variable: data_matrix})

    def transform(self, x):
        return self._woe_replace(x)

    def _woe_replace(self, x):
        x_bak = x.copy()
        for col in tqdm(x_bak.columns):
            dm = self.data_matrix[col]
            if col in self.continuous_col:
                x_bak.loc[x[col] == 'tails', col] = 0.0
                for i in range(len(dm)):
                    if np.isnan(dm['left'][i]):
                        x_bak.loc[x[col].isnull(), col] = dm['woe'][i]
                    else:
                        x_bak.loc[(x[col] >= dm['left'][i]) & (x[col] < dm['right'][i]), col] = dm['woe'][i]
        return x_bak

    def cal_bin_ks(self, x, y, oot=False, origin=False):
        df = pd.concat([x, pd.Series(y, name='y')], axis=1)
        new_data_matrix = {}
        if oot:
            print("cal bin ks, oot...")
            for col in tqdm(x.columns):
                if origin:
                    dm = self.data_matrix_origin[col].copy()
                else:
                    dm = self.data_matrix[col].copy()
                bad, obs, good, good_rate, bad_rate, ks = [], [], [], [], [], []
                for i in range(len(dm)):
                    if np.isnan(dm['left'][i]):
                        _x = df.loc[df[col].isnull(), col]
                        _y = df.loc[df[col].isnull(), 'y']
                    else:
                        _x = df.loc[(df[col] >= dm['left'][i]) & (df[col] < dm['right'][i]), col]
                        _y = df.loc[(df[col] >= dm['left'][i]) & (df[col] < dm['right'][i]), 'y']
                    bad.append(_y.sum())
                    obs.append(len(_x))
                    good.append(len(_x) - _y.sum())
                    good_rate.append((len(_x) - _y.sum()) / len(_x))
                    bad_rate.append(_y.sum() / len(_x))

                dm['bad'] = bad
                dm['obs'] = obs
                dm['good'] = good
                dm['good_rate'] = good_rate
                dm['bad_rate'] = bad_rate

                dm['prop_bad'] = dm['bad'] / sum(dm['bad'])
                dm['cum_prop_bad'] = dm['prop_bad'][0]
                for i in range(1, dm.shape[0]):
                    dm['cum_prop_bad'][i] = dm['cum_prop_bad'][i - 1] + dm['prop_bad'][i]
                dm['prop_good'] = dm['good'] / sum(dm['good'])
                dm['cum_prop_good'] = dm['prop_good'][0]
                for i in range(1, dm.shape[0]):
                    dm['cum_prop_good'][i] = dm['cum_prop_good'][i - 1] + dm['prop_good'][i]
                dm['ks'] = dm['cum_prop_good'] - dm['cum_prop_bad']
                dm['ks'] = round(dm['ks'], 4)
                dm = dm.drop(['prop_bad', 'cum_prop_bad', 'prop_good', 'cum_prop_good'], axis=1)
                self.calc_woe(dm, oot=oot)
                new_data_matrix[col] = dm
        else:
            print("cal bin ks, train...")
            for col in tqdm(x.columns):
                if origin:
                    dm = self.data_matrix_origin[col].copy()
                else:
                    dm = self.data_matrix[col].copy()
                dm['prop_bad'] = dm['bad'] / sum(dm['bad'])
                dm['cum_prop_bad'] = dm['prop_bad'][0]
                for i in range(1, dm.shape[0]):
                    dm['cum_prop_bad'][i] = dm['cum_prop_bad'][i - 1] + dm['prop_bad'][i]
                dm['prop_good'] = dm['good'] / sum(dm['good'])
                dm['cum_prop_good'] = dm['prop_good'][0]
                for i in range(1, dm.shape[0]):
                    dm['cum_prop_good'][i] = dm['cum_prop_good'][i - 1] + dm['prop_good'][i]
                dm['ks'] = dm['cum_prop_good'] - dm['cum_prop_bad']
                dm['ks'] = round(dm['ks'], 4)
                dm = dm.drop(['prop_bad', 'cum_prop_bad', 'prop_good', 'cum_prop_good'], axis=1)
                new_data_matrix[col] = dm
        return new_data_matrix

    def calc_woe(self, data, oot=False):
        """
        :param data: DataFrame(Var:float,bad:int,good:int)
        :return: weight of evidence
        """
        if 'bad' not in data.columns:
            raise ValueError("data columns don't has 'bad' column")
        if 'good' not in data.columns:
            raise ValueError("data columns don't has 'good' column")

        if oot:
            woe = np.log((data['bad'] / data['good']) / (data['bad'].sum() / data['good'].sum()))
            woe = woe.replace(-np.inf, self._WOE_MIN)
            woe = woe.replace(np.inf, self._WOE_MAX)
            data['iv'] = (data['bad'] / data['bad'].sum() - data['good'] / data['good'].sum()) * woe
        else:
            data['woe'] = np.log((data['bad'] / data['good']) / (data['bad'].sum() / data['good'].sum()))
            data['woe'] = data['woe'].replace(-np.inf, self._WOE_MIN)
            data['woe'] = data['woe'].replace(np.inf, self._WOE_MAX)
            data['iv'] = (data['bad'] / data['bad'].sum() - data['good'] / data['good'].sum()) * data['woe']
        return data[['woe', 'iv']]

    def cal_psi(self, data_matrix, data_matrix_oot):
        psi = {}
        for col in data_matrix:
            dm = data_matrix[col]
            dm_oot = data_matrix_oot[col]
            actual = dm['obs'] / dm['obs'].sum()
            expected = dm_oot['obs'] / dm_oot['obs'].sum()
            actual = actual.replace(0, 0.001)
            expected = expected.replace(0, 0.001)
            psi[col] = np.sum((actual - expected) * np.log(actual / expected))
        return psi
