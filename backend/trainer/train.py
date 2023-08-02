#!/usr/bin/env python
# coding: utf-8

# # Modeling

from operator import mod
from os import getcwd
from os.path import exists, join

import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
import random
import ast


from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

from sklearn.linear_model import LogisticRegression, LinearRegression
import warnings
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import  GradientBoostingClassifier

import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC, LinearSVC 
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import recall_score

from sklearn import tree
from sklearn.decomposition import PCA, SparsePCA

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import json
import pickle
from IPython.display import Image
import warnings
from sklearn.metrics import classification_report
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import RobustScaler, KBinsDiscretizer

from sklearn.base import TransformerMixin


from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from collections import Counter

from tqdm import tqdm

tqdm.pandas()


# Depression screener
dep_screener_cols = [
    'little_interest_in_doing_things',
    'feeling_down_depressed_hopeless',
    'trouble_falling_or_staying_asleep',
    'feeling_tired_or_having_little_energy',
    'poor_appetitie_or_overeating',
    'feeling_bad_about_yourself',
    'trouble_concentrating',
    'moving_or_speaking_to_slowly_or_fast',
    'thoughts_you_would_be_better_off_dead',
    'difficult_doing_daytoday_tasks'
]
model_features_opt2 = dep_screener_cols + [
    'seen_mental_health_professional',
    'times_with_12plus_alc',
    'time_since_last_healthcare',
    'cholesterol_prescription',
    'high_cholesterol',
    'age_in_years',
    'horomones_not_bc',
    'months_since_birth',
    'arthritis',
    'high_bp',
    'regular_periods',
    'moderate_recreation',
    'thyroid_issues',
    'vigorous_recreation',
    'stroke',
    'is_usa_born',
    'asthma',
    'count_days_moderate_recreational_activity',
    'have_health_insurance',
    'num_dep_screener_0',
    'weight_lbs_over_height_in_ratio'
]

model_features_low_opt7 = [
    'count_days_seen_doctor_12mo_bin',
    'times_with_12plus_alc',
    'seen_mental_health_professional',
    'count_lost_10plus_pounds',
    'arthritis',
    'horomones_not_bc',
    'is_usa_born',
    'times_with_8plus_alc',
    'time_since_last_healthcare',
    'duration_last_healthcare_visit',
    'work_schedule'
]

columns_to_use_low = model_features_low_opt7
columns_to_use_high = model_features_opt2


columns_to_use_high


for l in ['little_interest_in_doing_things',
 'feeling_down_depressed_hopeless',
 'trouble_falling_or_staying_asleep',
 'feeling_tired_or_having_little_energy',
 'poor_appetitie_or_overeating',
 'feeling_bad_about_yourself',
 'trouble_concentrating',
 'moving_or_speaking_to_slowly_or_fast',
 'thoughts_you_would_be_better_off_dead',
 'difficult_doing_daytoday_tasks',
 'seen_mental_health_professional',
 'times_with_12plus_alc',
 'time_since_last_healthcare',
 'cholesterol_prescription',
 'high_cholesterol',
 'age_in_years',
 'horomones_not_bc',
 'months_since_birth',
 'arthritis',
 'high_bp',
 'regular_periods',
 'moderate_recreation',
 'thyroid_issues',
 'vigorous_recreation',
 'stroke',
 'is_usa_born',
 'asthma',
 'count_days_moderate_recreational_activity',
 'have_health_insurance',
 'num_dep_screener_0',
 'weight_lbs_over_height_in_ratio']:
 print("'"+l+"'", ':', 1)
#  print(l, ': float | None = None')


len({'age_in_years': 0,
 'height_in': 48,
 'weight_lbs': 75,
 'is_usa_born': 1,
 'have_health_insurance': 1,
 'months_since_birth': 1,
 'regular_periods': 2,
 'horomones_not_bc': 2,
 'time_since_last_healthcare': 0,
 'seen_mental_health_professional': 1,
 'little_interest_in_doing_things': 3,
 'feeling_down_depressed_hopeless': 2,
 'trouble_falling_or_staying_asleep': 1,
 'feeling_tired_or_having_little_energy': 0,
 'poor_appetitie_or_overeating': 2,
 'feeling_bad_about_yourself': 2,
 'trouble_concentrating': 2,
 'moving_or_speaking_to_slowly_or_fast': 1,
 'thoughts_you_would_be_better_off_dead': 1,
 'difficult_doing_daytoday_tasks': 2,
 'times_with_12plus_alc': 2,
 'high_cholesterol': 1,
 'cholesterol_prescription': 2,
 'high_bp': 1,
 'moderate_recreation': 1,
 'count_days_moderate_recreational_activity': 4,
 'vigorous_recreation': 1,
 'thyroid_issues': 2,
 'arthritis': 2,
 'stroke': 1,
 'asthma': 1})


# # Opt 9: Ensemble Model
# 
# Build 2 models with different feature set
# - Model 1: 
#  - GB trained on observations with 1+ dep screener response. 
#  - Uses features from opt 2. 
#  - Uses undersampler.
# - Model 2: 
#  - RF trained on observations with 0 dep screener response. 
#  - Uses features from opt 7. 
#  - Uses undersampler
# 
# Notes
# - _low = has 9+ dep screeners answered 0
# - _high = has <9 dep screeners answered 0

len(list(set([model_features_low_opt7 + model_features_opt2][0])))


# class CustomFeatures(TransformerMixin):
#     def __init__(self, some_stuff=None, column_names= []):
#         pass
#         # self.some_stuff = some_stuff
#         self.column_names = column_names
#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         # do stuff on X, and return dataframe
#         # of the same shape - this gets messy
#         # if the preceding item is a numpy array
#         # and not a dataframe
#         if isinstance(X, np.ndarray):
#             X = pd.DataFrame(X)
#         X['num_dep_screener_0'] = (X[dep_screener_cols]==0).sum(axis=1)
#         X['weight_lbs_over_height_in_ratio'] = round(X['weight_lbs'] / X['height_in'],1)

#         return X


# # # using this by itself works as well
# # my_pipeline = make_pipeline(CustomFeatures(column_names=["my_str", "val"]))
# # my_pipeline.fit_transform(cdc_survey_pmom)


class CustomBin(TransformerMixin):
    def __init__(self, some_stuff=None, column_names= []):
        pass
        # self.some_stuff = some_stuff
        self.column_names = column_names
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # do stuff on X, and return dataframe
        # of the same shape - this gets messy
        # if the preceding item is a numpy array
        # and not a dataframe
        # if isinstance(X, np.ndarray):
        #     X = pd.DataFrame(X, columns=self.column_names)
        feature_values = X.dropna().values
        feature_values = feature_values.reshape([feature_values.shape[0],1])

        # create bins using estimator
        est = KBinsDiscretizer(
            n_bins=10,
            encode='ordinal', 
            strategy='uniform', 
            subsample=None
        )
        est.fit(feature_values)
        feature_values = est.transform(feature_values)

        fill_arr = X['count_days_seen_doctor_12mo'].values.copy()
        fill_arr[~np.isnan(fill_arr)] = np.asarray([val[0] for val in feature_values])
        X['count_days_seen_doctor_12mo_bin'] = fill_arr

        return X


# # using this by itself works as well
# my_pipeline = make_pipeline(CustomBin(column_names=["my_str", "val"]))
# my_pipeline.fit_transform(cdc_survey_pmom)


# v2 csv
df_cdc_clean = pd.read_csv('../../data/cdc_nhanes_survey_responses_clean.csv')

# filter to pregnant moms
cdc_survey_pmom = df_cdc_clean[df_cdc_clean['has_been_pregnant'] == 1]
print(cdc_survey_pmom.shape)

# add features
cdc_survey_pmom['num_dep_screener_0'] = (cdc_survey_pmom[dep_screener_cols]==0).sum(axis=1)
cdc_survey_pmom['weight_lbs_over_height_in_ratio'] = round(df_cdc_clean['weight_lbs'] / cdc_survey_pmom['height_in'],1)

feature_values = cdc_survey_pmom['count_days_seen_doctor_12mo'].dropna().values
feature_values = feature_values.reshape([feature_values.shape[0],1])

# create bins using estimator
est = KBinsDiscretizer(
    n_bins=10,
    encode='ordinal', 
    strategy='uniform', 
    subsample=None
)
est.fit(feature_values)
feature_values = est.transform(feature_values)

fill_arr = cdc_survey_pmom['count_days_seen_doctor_12mo'].values.copy()
fill_arr[~np.isnan(fill_arr)] = np.asarray([val[0] for val in feature_values])
cdc_survey_pmom['count_days_seen_doctor_12mo_bin'] = fill_arr





# subset to features and do preprocessing
data_low_dep_screener = cdc_survey_pmom[cdc_survey_pmom['num_dep_screener_0'] >= 9].copy()
data_low_dep_screener = data_low_dep_screener[['MDD'] + columns_to_use_low]
y_low = data_low_dep_screener['MDD'].values
x_low = data_low_dep_screener.iloc[:,1:].values

x_train_low, x_test_low, y_train_low, y_test_low = train_test_split(
    x_low, 
    y_low, 
    test_size=0.2, 
    random_state=42
)

# impute and scale
imputer_low = SimpleImputer(strategy='median')  
trans_low = RobustScaler()
x_train_low = imputer_low.fit_transform(x_train_low)
x_train_low = trans_low.fit_transform(x_train_low)
x_test_low = imputer_low.fit_transform(x_test_low)
x_test_low = trans_low.fit_transform(x_test_low)

# partially correct for class imbalance
rus = RandomUnderSampler(
    random_state=42, 
    sampling_strategy=0.12,
    replacement=False
)
x_train_low_rus, y_train_low_rus = rus.fit_resample(x_train_low,y_train_low)
print(f"x_train_low_rus: {x_train_low_rus.shape}")


# fit
rf = RandomForestClassifier(random_state=42)
rf.fit(x_train_low_rus, y_train_low_rus)
y_pred_low = rf.predict(x_test_low)

pd.DataFrame(classification_report(y_test_low,y_pred_low,output_dict=True))


model_filename = "model_pipeline_low.pkl"
model_path = join(getcwd(), model_filename)
joblib.dump(rf, model_path)



data_high_dep_screener = cdc_survey_pmom[cdc_survey_pmom['num_dep_screener_0'] < 9].copy()
data_high_dep_screener = data_high_dep_screener[['MDD'] + columns_to_use_high]
y_high = data_high_dep_screener['MDD'].values
x_high = data_high_dep_screener.iloc[:,1:].values

x_train_high, x_test_high, y_train_high, y_test_high = train_test_split(
    x_high, 
    y_high, 
    test_size=0.2, 
    random_state=42
) 
# impute and scale
imputer_high = SimpleImputer(strategy='median')  
trans_high = RobustScaler()
x_train_high = imputer_high.fit_transform(x_train_high)
x_train_high = trans_high.fit_transform(x_train_high)
x_test_high = imputer_high.fit_transform(x_test_high)
x_test_high = trans_high.fit_transform(x_test_high)

# partially correct for class imbalance
rus_model1 = RandomUnderSampler(
    random_state=42, 
    sampling_strategy=1,
    replacement=False
)
x_train_high_rus, y_train_high_rus = rus_model1.fit_resample(x_train_high,y_train_high)
print(f"x_train_high_rus: {x_train_high_rus.shape}")






gb = GradientBoostingClassifier(random_state=42)
gb.fit(x_train_high_rus, y_train_high_rus)
y_pred_high = gb.predict(x_test_high)

pd.DataFrame(classification_report(y_test_high,y_pred_high,output_dict=True))


model_filename = "model_pipeline_high.pkl"
model_path = join(getcwd(), model_filename)
joblib.dump(gb, model_path)



try:
    get_ipython().system('rm trainer.py')
except:
    pass


try:
    get_ipython().system('jupyter nbconvert --no-prompt --to script trainer.ipynb')
except:
    pass







