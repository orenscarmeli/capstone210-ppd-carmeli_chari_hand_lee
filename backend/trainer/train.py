#!/usr/bin/env python
# coding: utf-8

# # Modeling

# In[3]:


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


# In[4]:


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


# In[5]:


from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from collections import Counter

from tqdm import tqdm

tqdm.pandas()


# In[7]:


# v2 csv
df_cdc_clean = pd.read_csv('../data/cdc_nhanes_survey_responses_clean.csv')

# filter to moms
df_cdc_clean = df_cdc_clean[df_cdc_clean['has_been_pregnant'] == 1]
df_cdc_clean = df_cdc_clean.drop(columns=['has_been_pregnant'])

df_cdc_clean


# In[8]:


cols_to_keep = [
    'SEQN',
    'MDD',
    # 'is_male',
    # 'has_been_pregnant',
    'age_with_angina_pectoris',
    'age_liver_condition',
    'age_range_first_menstrual_period',
    'annual_healthcare_visit_count',
    'have_liver_condition',
    'type_of_work_done_last_week',
    'weight_change_intentional',
    'days_nicotine_substitute_used',
    'pain_relief_from_cardio_recoverytime',
    # Depression screener
    'little_interest_in_doing_things',
    'feeling_down_depressed_hopeless',
    'trouble_falling_or_staying_asleep',
    'feeling_tired_or_having_little_energy',
    'poor_appetitie_or_overeating',
    'feeling_bad_about_yourself',
    'trouble_concentrating',
    'moving_or_speaking_to_slowly_or_fast',
    'thoughts_you_would_be_better_off_dead',
    'difficult_doing_daytoday_tasks',
    # Alcohol & smoking
    'has_smoked_tabacco_last_5days',
    'alcoholic_drinks_past_12mo',    
    # Diet & Nutrition
    'how_healthy_is_your_diet',    
    'count_lost_10plus_pounds',
    'has_tried_to_lose_weight_12mo',       
    # Physical health & Medical History
    'count_days_seen_doctor_12mo',
    'duration_last_healthcare_visit',        
    'count_days_moderate_recreational_activity',   
    'count_minutes_moderate_recreational_activity',
    'count_minutes_moderate_sedentary_activity',
    'general_health_condition',    
    'has_diabetes',
    'has_overweight_diagnosis',         
    # Demographic data
    'food_security_level_household',   
    'food_security_level_adult',    
    'monthly_poverty_index_category',
    'monthly_poverty_index',
    'count_hours_worked_last_week',
    'age_in_years',   
    'education_level',
    'is_usa_born',    
    'has_health_insurance',
    'has_health_insurance_gap'   
]
len(cols_to_keep)


# In[9]:


df_cdc_clean = df_cdc_clean[cols_to_keep]
df_cdc_clean


# In[10]:


keep_feats = [
    'has_health_insurance',
    'difficult_doing_daytoday_tasks',
    'age_range_first_menstrual_period',
    'weight_change_intentional',
    'thoughts_you_would_be_better_off_dead',
    'little_interest_in_doing_things',
    'trouble_concentrating',
    'food_security_level_household',
    'general_health_condition',
    'monthly_poverty_index',
    'food_security_level_adult',
    'count_days_seen_doctor_12mo',
    'has_overweight_diagnosis',
    'feeling_down_depressed_hopeless',
    'count_minutes_moderate_recreational_activity',
    'have_liver_condition',
    'pain_relief_from_cardio_recoverytime',
    'education_level',
    'count_hours_worked_last_week',
    'age_in_years',
    'has_diabetes',
    'alcoholic_drinks_past_12mo',
    'count_lost_10plus_pounds',
    'days_nicotine_substitute_used',
    'age_with_angina_pectoris',
    'annual_healthcare_visit_count',
    'poor_appetitie_or_overeating',
    'feeling_bad_about_yourself',
    'has_tried_to_lose_weight_12mo',
    'count_days_moderate_recreational_activity',
    'count_minutes_moderate_sedentary_activity'
]
len(keep_feats)


# In[11]:


# SEQN and MDD are the first two columns in the df, so exclude from X
X = df_cdc_clean.iloc[:,2:].values
y = df_cdc_clean['MDD'].values


# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )


# In[13]:


algo_name = 'Logistic Regression'

X = df_cdc_clean[keep_feats]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

my_imputer = SimpleImputer()
X_train = my_imputer.fit_transform(X_train)
X_test = my_imputer.fit_transform(X_test)

sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

n_samples, n_features = X.shape

# Model Pipeline
processing_pipeline = make_pipeline(SimpleImputer(), MinMaxScaler(), LogisticRegression(max_iter=1000, penalty='l2', C=10))


model_filename = "model_pipeline.pkl"
model_path = join(getcwd(), model_filename)
if not exists(model_path):
    processing_pipeline.fit(X_train.values, y_train)
    pred_labels  = processing_pipeline.predict(X_test.values)
    pred_labels = [x.round() for x in pred_labels]

    joblib.dump(processing_pipeline, model_path)
    print('successfully trained model')
else:
    print("Model has already been trained, no need to rerun")


# In[ ]:


try:
    get_ipython().system('rm trainer.py')
except:
    pass


# In[ ]:


try:
    get_ipython().system('jupyter nbconvert --no-prompt --to script trainer.ipynb')
except:
    pass


