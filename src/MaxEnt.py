# -*- coding: utf-8 -*-
# @Time    : 2019-04-16 16:39
# @Author  : finupgroup
# @FileName: MaxEnt.py
# @Software: PyCharm

import math
from collections import defaultdict


class MaxEnt(object):
    def init_params(self, X, Y):
        self.X_ = X
        self.Y_ = set()

        self.cal_Pxy_Px(X, Y)

        self.N = len(X)                 # 训练集大小
        self.n = len(self.Pxy)          # 书中(x,y)对数
        self.M = 10000.0                # 书91页那个M，但实际操作中并没有用那个值
        # 可认为是学习速率

        self.build_dict()
        self.cal_EPxy()

    def build_dict(self):
        self.id2xy = {}
        self.xy2id = {}

        for i, (x, y) in enumerate(self.Pxy):
            self.id2xy[i] = (x, y)
            self.xy2id[(x, y)] = i

    def cal_Pxy_Px(self, X, Y):
        self.Pxy = defaultdict(int)
        self.Px = defaultdict(int)

        for i in range(len(X)):
            x_, y = X[i], Y[i]
            self.Y_.add(y)

            for x in x_:
                self.Pxy[(x, y)] += 1
                self.Px[x] += 1

    def cal_EPxy(self):
        """
        计算书中82页最下面那个期望
        """
        self.EPxy = defaultdict(float)
        for id in range(self.n):
            (x, y) = self.id2xy[id]
            self.EPxy[id] = float(self.Pxy[(x, y)]) / float(self.N)

    def cal_pyx(self, X, y):
        result = 0.0
        for x in X:
            if self.fxy(x, y):
                id = self.xy2id[(x, y)]
                result += self.w[id]
        return (math.exp(result), y)

    def cal_probality(self, X):
        """
        计算书85页公式6.22
        """
        Pyxs = [(self.cal_pyx(X, y)) for y in self.Y_]
        Z = sum([prob for prob, y in Pyxs])
        return [(prob / Z, y) for prob, y in Pyxs]

    def cal_EPx(self):
        """
        计算书83页最上面那个期望
        """
        self.EPx = [0.0 for i in range(self.n)]

        for i, X in enumerate(self.X_):
            Pyxs = self.cal_probality(X)

            for x in X:
                for Pyx, y in Pyxs:
                    if self.fxy(x, y):
                        id = self.xy2id[(x, y)]

                        self.EPx[id] += Pyx * (1.0 / self.N)

    def fxy(self, x, y):
        return (x, y) in self.xy2id

    def train(self, X, Y):
        self.init_params(X, Y)
        self.w = [0.0 for i in range(self.n)]

        max_iteration = 1000
        for times in range(max_iteration):
            print('iterater times %d' % times)
            sigmas = []
            self.cal_EPx()

            for i in range(self.n):
                sigma = 1 / self.M * math.log(self.EPxy[i] / self.EPx[i])
                sigmas.append(sigma)

            # if len(filter(lambda x: abs(x) >= 0.01, sigmas)) == 0:
            #     break

            self.w = [self.w[i] + sigmas[i] for i in range(self.n)]

    def predict(self, testset):
        results = []
        for test in testset:
            result = self.cal_probality(test)
            results.append(result[1][0])
        return results


def rebuild_features(features):
    """
    将原feature的（a0,a1,a2,a3,a4,...）
    变成 (0_a0,1_a1,2_a2,3_a3,4_a4,...)形式
    """
    new_features = []
    for feature in features:
        new_feature = []
        for i, f in enumerate(feature):
            new_feature.append(str(i) + '_' + str(f))
        new_features.append(new_feature)
    return new_features


