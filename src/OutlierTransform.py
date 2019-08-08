# -*- coding: utf-8 -*-
# @Time    : 2019-04-30 11:35
# @Author  : finupgroup
# @FileName: OutlierTransform.py
# @Software: PyCharm
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class OutlierTransform:
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

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
