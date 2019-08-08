# -*- coding: utf-8 -*-
# @Time    : 2019-04-17 09:55
# @Author  : finupgroup
# @FileName: CategoricalTransform.py
# @Software: PyCharm

import numpy as np


class CategoricalTransform:
    """

    bad_rate替换
    """
    def __init__(self):
        self.categorical_var = []
        self.dictionary = dict()

    def fit(self, data, categorical_variance, y):
        """
        分类变量进行连续化变换 bad_rate替换
        :param data: pd.DataFrame columns variable
        :param y: pd.Series label
        :param categorical_variance: categorical_variance
        :return: CategoricalTransform对象
        """

        if len(categorical_variance) > 0:
            data_cate = data.loc[:, categorical_variance]
            nan_rate = data_cate.apply(self._nan_transform, axis=0)
            data_cate["label"] = y
            for elem in categorical_variance:
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
