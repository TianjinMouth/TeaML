[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![GitHub top language](https://img.shields.io/github/languages/top/didi/delta)](https://img.shields.io/github/languages/top/didi/delta)
[![GitHub Issues](https://img.shields.io/github/issues/didi/delta.svg)](https://github.com/didi/delta/issues)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/didi/delta/blob/master/LICENSE)

# **Tea**

🎉🎉🎉 We are proud to announce that we design an automatic modeling robot based on `financial risk control field`! 🎉🎉🎉

## Overview

Tea is a simple and design friendly automatic modeling learning framework.
It can automatically model from beginning to end, and in the end, it will also help you output a model report about the model.

- **Human-friendly**. Tea's code is straightforward, well documented and tested, which makes it very easy to understand and modify.
- **Built-in financial risk control field**. Tea built-in financial risk control field, it fits well with the use in the field of financial risk control, including WOE, and is very suitable for this scenario.
- **Flexible**. Tea provides a variety of variable selection methods, each of which can be self-defined. You can also assemble these algorithms in different order. 
- **Final Report**. Tea can provide you with a final version of the model report, so that you can find the details in your model. 

## Our Goal

- **Automation** In the near future, we will update and add some fantastic algorithms, including but not limited to variable generation (VariableCluster is already in experimental function).
- **Common Use** All algorithmic engineers, including model analysts, can use it to increase efficiency as long as you have some algorithmic knowledge.
- **Wonderful thing** We hope that there will be many wonderful things to add. At present, there is no optimization algorithm and parallel strategy in this version. We will try to add these things in later iterations, maybe not too long.

## Performance

| Task                                        | Strategy | Dataset                   | Score           | Detail                                                                                                             |
| ------------------------------------------- | -------- | ------------------------- | --------------- | -------------------------- |
| Predicting the Delay Rate of Financial Risk |   Tea    | Financial Risk Data       | **0.6894** (AUC) | WOE(Monotonic) + STEPWISE |
| Predicting the Delay Rate of Financial Risk | LightGBM | Financial Risk Data       | **0.6773** (AUC) |         LightGBM          |


## Quick start

### Requirements and Installation


The project is based on Python 3.7, Python 3.6 may also work, but it is not fully tested to ensure that all functions are normal.

```bash
pip install Tea
```

### Example Usage

Let's run a simple version.

```python
from Tea.utils.tea_encoder import *
from Tea.utils.tea_filter import *
from Tea.utils.tea_utils import *
from Tea.utils.auto_bin_woe import *
import Tea

data = pd.read_csv("Tea/examples.csv")

# encoder
ct = TeaBadRateEncoder(num=1)
me = TeaMeanEncoder(categorical_features=['city'])
t = TeaOneHotEncoder()
encoder = [me]

# woe & feature selection
woe = Tea.WOE(bins=10, bad_rate_merge=True, bad_rate_sim_threshold=0.05, psi_threshold=0.1, iv_threshold=None)
iv = FilterIV(200, 100)
vif = FilterVif(50)
mod = FilterModel('lr', 70)
nova = FilterANOVA(40, 30)
coline = FilterCoLine({'penalty': 'l2', 'C': 0.01, 'fit_intercept': True})
fshap = FilterSHAP(70)
outlier = OutlierTransform()
filtercor = FilterCorr(20)
stepwise = FilterStepWise(method='p_value')
method = [woe, stepwise]

# main
tea = Tea.Tea(['core_lend_request_id', 'lend_customer_id', 'customer_sex',
               'data_center_id', 'trace_back_time', 'mobile', 'user_id', 'id_no', 'task_id', 'id',
               'id_district_name', 'id_province_name', 'id_city_name', 'pass_time'],
              'is_overdue_M0',
              datetime_feature='pass_time',
              split_method='oot',
              file_path='report.xlsx')
tea.wash(data, null_drop_rate=0.8, zero_drop_rate=0.9)
tea.cook(encoder)
tea.select(method)
tea.drink(LogisticRegression(penalty='l2', C=1, class_weight='balanced'))
tea.sleep(woe.bins)


'''
Preliminary screening...
100%|██████████| 19/19 [00:00<00:00, 29.19it/s]
100%|██████████| 19/19 [00:00<00:00, 50.03it/s]
100%|██████████| 19/19 [00:00<00:00, 55.00it/s]
100%|██████████| 19/19 [00:00<00:00, 104.02it/s]
  0%|          | 0/19 [00:00<?, ?it/s]
cal bin ks, train...
cal bin ks, oot...
100%|██████████| 19/19 [00:00<00:00, 21.33it/s]
100%|██████████| 100/100 [00:00<00:00, 116.78it/s]
  0%|          | 0/19 [00:00<?, ?it/s]
Train AUC: 0.6107958854166341
Test AUC: 0.6083763215945612
OOT AUC: 0.6050562520208106
Train KS: 0.1719605325145203
Test KS: 0.17401800497420833
OOT KS: 0.1616794283922675
--------------------------------------------------- 

cal bin ks, train...
100%|██████████| 19/19 [00:00<00:00, 97.39it/s]
 36%|███▋      | 4/19 [00:00<00:00, 37.62it/s]
cal bin ks, oot...

100%|██████████| 19/19 [00:00<00:00, 25.66it/s]
Add  P                              with p-value 1.14745e-22
Add  F                              with p-value 3.38993e-15
Add  I                              with p-value 5.18381e-10
Add  J                              with p-value 2.8625e-09
Add  M                              with p-value 6.66696e-07
Add  Q                              with p-value 3.18125e-09
Add  B                              with p-value 1.14541e-06
Add  D                              with p-value 1.21802e-05
Add  K                              with p-value 2.70815e-05
Add  C                              with p-value 0.000118247
Add  A                              with p-value 0.000214666
Add  L                              with p-value 0.000169921
Add  H                              with p-value 0.00139263
Add  N                              with p-value 0.000488745

   feature_name  feature_coef
5             Q      2.332818
10            A      2.203708
12            H      1.391547
7             D      1.385142
2             I      1.192397
13            N      1.181320
0             P      0.926443
8             K      0.914186
1             F      0.898581
3             J      0.868826
6             B      0.864311
4             M      0.851936
11            L      0.842446
9             C      0.704460
Finish 🍵 

'''
```

#### What's the encoder in tea.cook()？

This is a module for automatic processing of discrete variables in robots.

We offer you three ways to deal with categorical variables

```python
ct = TeaBadRateEncoder(num=1)
me = TeaMeanEncoder(categorical_features=['city'])
t = TeaOneHotEncoder()
encoder = [me]
```


```TeaBadRateEncoder:  Replace categorical variables with bad_rate of each bin```

```TeaMeanEncoder:  MeanEncoder```

```TeaOneHotEncoder: Such as Onehot```



#### What's the method in tea.cook()

This is a module for automatic selection of variables in robots.

What you fill in in the Tea's method is orderly.

For example, the following represents a monotone woe transformation of all variables, followed by a step-by-step regression based on p-value.

```python
woe = Tea.WOE(bins=10, bad_rate_merge=True, bad_rate_sim_threshold=0.05, psi_threshold=0.1, iv_threshold=None)
iv = FilterIV(200, 100)
vif = FilterVif(50)
mod = FilterModel('lr', 70)
nova = FilterANOVA(40, 30)
coline = FilterCoLine({'penalty': 'l2', 'C': 0.01, 'fit_intercept': True})
fshap = FilterSHAP(70)
outlier = OutlierTransform()
filtercor = FilterCorr(20)
stepwise = FilterStepWise(method='p_value')
method = [woe, stepwise]
```

## Support

We support all people to make suggestions, because this is support and encouragement for our project.

