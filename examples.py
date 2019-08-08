from Tea import *
test = pd.read_csv('test.csv')

# encoder
ct = TeaBadRateEncoder(num=1)
me = TeaMeanEncoder(categorical_features=[])
t = TeaOneHotEncoder()
encoder = [me]

# woe & feature selection
woe = WOE(bins=10, bad_rate_merge=True, bad_rate_sim_thres=0.05, psi_threshold=0.1, iv_threshold=None)
iv = FilterIV(200, 100)
vif = FilterVif(50)
mod = FilterModel('lr', 70)
nova = FilterANOVA(40, 30)
coline = FilterCoLine({'penalty': 'l2', 'C': 0.01, 'fit_intercept': True})
fshap = FilterSHAP(70)
outlier = OutlierTransform()
filtercor = FilterCorr(20)
stepwise = FilterStepWise(method='p_value')
method = [woe]

# main
tea = Tea(['core_lend_request_id', 'lend_customer_id', 'customer_sex', 'data_center_id', 'trace_back_time',
           'mobile', 'is_overdue_M1', 'user_id', 'id_no', 'task_id', 'id', 'id_district_name',
           'id_province_name', 'id_city_name'],
          'is_overdue_M0',
          datetime_feature='pass_time',
          oot_threshold='2019-01-01',
          null_drop_rate=0.8,
          zero_drop_rate=0.9)
tea.wash(test)
tea.cook(encoder)
tea.select(method)
tea.drink(LogisticRegression(penalty='l2', C=1, class_weight='balanced'))
tea.sleep(woe.bins)
