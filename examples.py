# from Tea.utils.tea_encoder import *
# from Tea.utils.tea_filter import *
# from Tea.utils.tea_utils import *
# from Tea.utils.auto_bin_woe import *
from Tea import *

data = pd.read_csv("/Users/finup/Project/LossRecontact/Data/past_record_features_all.csv")
data.columns = [c.replace('p.', '').replace('s.', '').replace('_u1.', '') for c in data.columns]
qz = data[(data['id_no'].notnull()) & (data['service_user_id']==4)].reset_index(drop=True)

# encoder
ct = TeaBadRateEncoder(num=1)
me = TeaMeanEncoder(categorical_features=[])
t = TeaOneHotEncoder()
encoder = []

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
method = []

# main
tea = Tea.Tea(['called_name', 'id_no', 'called_no', 'id_no_biz', 'called_no_biz', 'recent_ring_time', 'name',
              'etl_tx_dt', 'ring', 'service_user_id', 'is_roll_back', 'best_repay_status','connect_cnt',
              'close_relation', 'positive_respond'], 'is_roll_back', datetime_feature='ring', split_method='oos',
              file_path='/Users/finup/Desktop/past.xlsx')
tea.wash(qz, null_drop_rate=0.8, zero_drop_rate=0.9)
tea.cook(encoder)
tea.select(method)
tea.drink(LGBMClassifier(boosting_type='gbdt', class_weight=None,
        colsample_bytree=0.8774295384701779, importance_type='split',
        learning_rate=0.01, max_depth=-1, min_child_samples=170,
        min_child_weight=0.001, min_split_gain=0.0, n_estimators=60,
        n_jobs=-1, num_leaves=8, objective=None, random_state=None,
        reg_alpha=0.0, reg_lambda=10, silent=True, subsample=1.0,
        subsample_for_bin=200000, subsample_freq=0))
tea.sleep(woe.bins)
