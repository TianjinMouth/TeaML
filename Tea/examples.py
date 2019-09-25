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
