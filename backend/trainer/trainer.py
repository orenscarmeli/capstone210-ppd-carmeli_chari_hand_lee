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


from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from collections import Counter

from tqdm import tqdm

tqdm.pandas()


# df_cdc_clean = pd.read_csv('../data/cdc_nhanes_survey_responses_clean.csv')
# v2 csv
df_cdc_clean = pd.read_csv('../data/df_cdc_clean_v2.csv')

# filter to moms
df_cdc_clean = df_cdc_clean[df_cdc_clean['has_been_pregnant'] == 1]
df_cdc_clean = df_cdc_clean.drop(columns=['has_been_pregnant'])

df_cdc_clean


# cols_to_keep = ['SEQN', 'MDD']
# # cols_to_keep.extend(df_cdc_clean.columns.tolist()[-38:])
# cols_to_keep.extend([col for col in df_cdc_clean.columns.tolist() if '_' in col and '_x' not in col and '_y' not in col])
# cols_to_keep


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
len(all_columns)


df_cdc_clean = df_cdc_clean[cols_to_keep]
df_cdc_clean


# ProfileReport(df_cdc_clean, title="Profiling Report")


# SEQN and MDD are the first two columns in the df, so exclude from X
X = df_cdc_clean.iloc[:,2:].values
y = df_cdc_clean['MDD'].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


print(X_train.shape)
print(X_test.shape)

print(y_train.shape)
print(y_test.shape)


def get_classification_report(pred_labels, y_test, algo_name, show_full_report=False):
    eval_on = 'macro avg' 
    # eval_on = 'depressed' #not_depressed
    target_names = ['not_depressed', 'depressed',]
    
    df_cls_rpt = pd.DataFrame(
        classification_report(
            y_test, 
            pred_labels, 
            target_names=target_names, 
            output_dict=True
        )
    ).rename_axis('metric')\
    .reset_index()

    
    if show_full_report == True:
        display(df_cls_rpt)
    accuracy = df_cls_rpt[['accuracy']].iloc[0, 0]
    df_cls_rpt = df_cls_rpt[['metric', eval_on]].T
    df_cls_rpt.columns = df_cls_rpt.iloc[0,:]
    df_cls_rpt = df_cls_rpt.iloc[1:,:]
    df_cls_rpt['accuracy'] = accuracy

    df_cls_rpt['algo'] = algo_name

    first_column = df_cls_rpt.pop('algo')
    df_cls_rpt.insert(0, 'algo', first_column)

    # display(df_cls_rpt)
    return df_cls_rpt


def plot_confusion_matrix(y_test, pred_labels):
    """
    Function that displays a confusion matrix for provided true and predicted classes
    """
    #print(f'cover type 1 and type 2 total correct {np.sum(np.diag(metrics.confusion_matrix(y_test, pred_labels))[:2])}')

    cm = confusion_matrix(y_test, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(5,5))
    disp = disp.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='horizontal')    
    plt.grid(False)
    plt.show()
    return


def algo_grid_search_pred(
    X_train, 
    y_train, 
    X_test,
    y_test,
    algo,
    algo_name,
    params,
    cv,
    verbose,
    imputer,
    scaler):
    score_on = 'recall' #'f1_score'

    pipeline_list = [imputer, scaler, algo]
    # drop imputer or scaler if none
    pipeline_list = [step for step in pipeline_list if step is not None]

    processing_pipeline = make_pipeline(imputer, scaler, algo)
    processing_pipeline = make_pipeline()
    i = 1
    for step in pipeline_list:
        processing_pipeline.steps.append([(type(step).__name__).lower(), step])
        i += 1
    grid = GridSearchCV(
        processing_pipeline, 
        param_grid=params, 
        n_jobs=-1, 
        cv=cv, 
        verbose=verbose, 
        scoring='recall')
    grid.fit(X_train, y_train)

    pred_labels = [x.round() for x in grid.best_estimator_.predict(X_test)]
    pred_labels = [x.round() for x in pred_labels]

    df_algo_cls_rpt = get_classification_report(pred_labels, y_test, algo_name)
    df_algo_cls_rpt['train_r2'] = grid.best_estimator_.score(X_train, y_train)
    df_algo_cls_rpt['test_r2'] = grid.best_estimator_.score(X_test, y_test)
    df_algo_cls_rpt['best_params'] = str(grid.best_params_)
    tn, fp, fn, tp = confusion_matrix(pred_labels, y_test).ravel()
    df_algo_cls_rpt['tp'] = tp
    df_algo_cls_rpt['fn'] = fn
    df_algo_cls_rpt['fp'] = fp
    df_algo_cls_rpt['tn'] = tn

    
    return df_algo_cls_rpt, pred_labels


def algo_baseline_pred(
    X_train, 
    y_train, 
    X_test,
    y_test,
    show_full_report,
    algo,
    algo_name,
    imputer,
    scaler):

    pipeline_list = [imputer, scaler, algo]
    # drop imputer or scaler if none
    pipeline_list = [step for step in pipeline_list if step is not None]
    
    try:
        processing_pipeline = make_pipeline()
        i = 1
        for step in pipeline_list:
            processing_pipeline.steps.append([(type(step).__name__).lower(), step])
            i += 1

        processing_pipeline.fit(X_train, y_train)
        pred_labels  = processing_pipeline.predict(X_test)
        pred_labels = [x.round() for x in pred_labels]

        df_algo_cls_rpt = get_classification_report(pred_labels, y_test, algo_name, show_full_report)
        tn, fp, fn, tp = confusion_matrix(pred_labels, y_test).ravel()
        df_algo_cls_rpt['tp'] = tp
        df_algo_cls_rpt['fn'] = fn
        df_algo_cls_rpt['fp'] = fp
        df_algo_cls_rpt['tn'] = tn
        
        return df_algo_cls_rpt, pred_labels
    except ValueError as e:
        print(e)
        raise Exception (f'{algo_name} might not work with NaN')
        
    return
    


def baseline_models(
    X_train, 
    y_train, 
    X_test, 
    y_test,
    algo_attempt_list,
    do_smote=False,
    show_confusion_matrix=False,
    show_full_report=False,
    grid_search=False,
    cv=5,
    verbose=0,
    imputer=SimpleImputer(),
    scaler=RobustScaler()
    ):
    """

    """



    # do_smote
    if do_smote == True:
        # have to impute first because smote won't take nulls
        my_imputer = SimpleImputer()
        X_train = my_imputer.fit_transform(X_train)
        X_test = my_imputer.fit_transform(X_test)

        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)


    df_cls_rpt = pd.DataFrame()
    conf_mtrx_dict = {}

    if algo_attempt_list == 'all':
        algo_attempt_list = ['knn', 'lm', 'bnb', 'gnb', 'dt', 'rf', 'gb', 'xgb']
    
    for a in algo_attempt_list:
        # SVR
        if a == 'svr':
            algo_name = 'SVR'
            params = {
                "simpleimputer__strategy": ["mean", "median"],
                "robustscaler__quantile_range": [(25.0, 75.0), (30.0, 70.0)],
                "svr__C": [0.1, 1.0],
                "svr__gamma": ["auto", 0.1],
            }
            algo = SVR()
        elif a == 'knn':
            algo_name = 'KNN'
            params = {
                'kneighborsclassifier__n_neighbors': list(range(1, 15))
            }
            print(params)
            algo = KNeighborsClassifier()
        elif a == 'lm':
            algo_name = 'Logistic Regression'
            params = {
                'logisticregression__penalty': ['l1','l2'], 
                'logisticregression__C': [0.001,0.01,0.1,1,10,100,1000]
            }
            algo = LogisticRegression(max_iter=1000, penalty='l2', C=10)
        elif a == 'bnb':
            algo_name = 'Bernoulli Naive Bayes'
            params = {
                'bernoullinb__alpha': np.logspace(0,-9, num=100),
                'bernoullinb__binarize': [0.0, 1.0, 2.0],
                # 'bernoullinb__fit_prior': [True, False]
            }
            algo = BernoulliNB()
        elif a == 'gnb':
            algo_name = 'Gaussian Naive Bayes'
            params = {
                'gaussiannb__var_smoothing': np.logspace(0,-9, num=100)
            }
            algo = GaussianNB()
        elif a == 'dt':
            algo_name = 'Decision Tree'
            params = {
                'decisiontreeclassifier__criterion':['gini', 'entropy', 'logloss'],
                'decisiontreeclassifier__max_depth': np.arange(1, 15)
            }
            algo = DecisionTreeClassifier(random_state=42)
        elif a == 'rf':
            algo_name = 'Random Forest'
            params = {
                'randomforestclassifier__criterion': ['gini', 'entropy', 'logloss'],
                "randomforestclassifier__n_estimators": [10, 20, 40, 80, 100, 125, 150],
                "randomforestclassifier__max_features": ["sqrt", "log2", None],
                "randomforestclassifier__min_samples_split": [1, 2, 4, 8],
                "randomforestclassifier__bootstrap": [True, False],
            }
            algo = RandomForestClassifier(random_state=42)
        elif a == 'gb':
            algo_name = 'Gradient Boosting Classifier'
            params = {
                "gradientboostingclassifier__loss":["log_loss", "exponential"],
                "gradientboostingclassifier__learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
                # "gradientboostingclassifier__min_samples_split": np.linspace(0.1, 0.5, 12),
                # "gradientboostingclassifier__min_samples_leaf": np.linspace(0.1, 0.5, 12),
                "gradientboostingclassifier__max_depth":[3,5,8],
                "gradientboostingclassifier__max_features":["log2", "sqrt"],
                "gradientboostingclassifier__criterion": ["friedman_mse", "mae"],
                # "gradientboostingclassifier__subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
                "gradientboostingclassifier__n_estimators":[10, 25, 50, 100, 125, 150]
            }
            algo = GradientBoostingClassifier()
        elif a == 'xgb':
            algo_name = 'XGBoost'
            params = {
                'xgbclassifier__max_depth': [3, 4, 5, 6, 8, 10, 12, 15],
                'xgbclassifier__learning_rate': [0.001, 0.01, 0.1, 0.20, 0.25, 0.30],
                "xgbclassifier__gamma":[0, 0.25, 0.5, 0.75,1],
                'xgbclassifier__n_estimators': [100, 500, 1000],
            }
            algo = xgb.XGBClassifier()
        else:
            raise Exception(f'{a} is not a supported algorithm')
            # print(f'{a} is not a supported algorithm')

        if grid_search == True and params is not None:
            df_algo_cls_rpt, pred_labels = algo_grid_search_pred(
                X_train, 
                y_train, 
                X_test,
                y_test,
                algo=algo,
                algo_name=algo_name,
                params=params,
                imputer=imputer,
                scaler=scaler,
                cv=cv,
                verbose=verbose
            )
            
        else:
            if grid_search == True and params is None:
                # print(f'Params not defined for {algo_name} GridSearch. Fitting baseline model.')
                pass
            df_algo_cls_rpt, pred_labels = algo_baseline_pred(
                X_train, 
                y_train, 
                X_test,
                y_test,
                show_full_report,
                algo=algo,
                algo_name=algo_name,
                imputer=imputer,
                scaler=scaler,
            )

        df_cls_rpt = pd.concat([
                df_algo_cls_rpt, 
                df_cls_rpt], 
                ignore_index=True
        )
        conf_mtrx_dict[algo_name] = pred_labels
    

    if show_confusion_matrix:
        for k,v in conf_mtrx_dict.items():
            print(f'{k} Confusion Matrix')
            plot_confusion_matrix(y_test, v)

    df_cls_rpt.sort_values(by=['f1-score'], ascending=False, inplace=True)
    df_cls_rpt = df_cls_rpt.reset_index(drop=True)
    return df_cls_rpt



baseline_models(
    X_train, 
    y_train, 
    X_test, 
    y_test, 
    algo_attempt_list=['bnb',],
    grid_search=True,
    cv=3,
    verbose=0,
    imputer=SimpleImputer()
    )


# # Brute Force 
# Can select different algo, imputer, scaler, etc

# import warnings
# warnings.filterwarnings("ignore")


# ##### with all setting each trial takes a while ##########
# ##### commenting out options can make bring iteration time down drastically ###########
# num_trials = 2
# best_score = 0
# primary_eval_metric = 'recall'
# secondary_eval_metric = 'f1-score'
# best_cols = []
# df_best_scores = pd.DataFrame()

# try:
    
#     for i in tqdm(range(1, num_trials+1)):
#         df_random_feats = df_cdc_clean.iloc[:,2:]
#         rand_num_features = random.randint(3, (df_random_feats.shape[1]))
#         df_random_feats = df_random_feats.sample(n=rand_num_features, axis='columns')
#         X = df_random_feats

#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42
#         )

#         # SMOTE or not
#         for b in [True, False]:
#             # different imputers
#             for imp in [KNNImputer(), SimpleImputer(strategy='mean'), SimpleImputer(strategy='most_frequent')]:
#                 for scl in [RobustScaler(), StandardScaler(), MinMaxScaler(), Normalizer()]:
#                     df_scores = baseline_models(
#                         X_train, 
#                         y_train, 
#                         X_test, 
#                         y_test, 
#                         do_smote=b, 
#                         algo_attempt_list=['bnb', 'gnb', 'lm'],
#                         grid_search=True,
#                         cv=5,
#                         verbose=0,
#                         imputer=imp,
#                         scaler=scl
#                     )
#                     df_scores['SMOTE'] = b
#                     df_scores['imputer'] = imp
#                     df_scores['scaler'] = scl
#                     # sort to keep best
#                     df_scores.sort_values(
#                         by=[primary_eval_metric, secondary_eval_metric], 
#                         inplace=True, 
#                         ascending=False
#                     )
#                     df_scores = df_scores.reset_index()
#                     df_scores = df_scores.iloc[:,1:]
#                     if df_scores[eval_metric][0] > best_score:
#                         best_f1 = df_scores[eval_metric][0]
#                         best_cols = df_random_feats.columns
#                         df_scores['features'] = str(df_random_feats.columns.tolist())
#                         df_scores['trial_num'] = i
#                         df_best_scores = pd.concat([df_best_scores, df_scores.iloc[0:1,:]], ignore_index=True)
# except KeyboardInterrupt as e:
#     print(e)

# df_best_scores.sort_values(
#     by=[primary_eval_metric, secondary_eval_metric], 
#     inplace=True, 
#     ascending=False
# )
# df_best_scores = df_best_scores.reset_index(drop=True)
# print(best_f1)
# print(best_cols)
# df_best_scores


# ## Randomized choices for imputer, SMOTE, and scaler

import warnings
warnings.filterwarnings("ignore")


##### with all setting each trial takes a while ##########
##### commenting out options can make bring iteration time down drastically ###########
num_trials = 30
best_score = 0
primary_eval_metric = 'f1-score'
secondary_eval_metric = 'recall' 

best_cols = []
df_best_scores = pd.DataFrame()

try:
    
    for i in tqdm(range(1, num_trials+1)):
        df_random_feats = df_cdc_clean.iloc[:,2:]
        rand_num_features = random.randint(3, (df_random_feats.shape[1]))
        df_random_feats = df_random_feats.sample(n=rand_num_features, axis='columns')
        X = df_random_feats

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # SMOTE or not
        for b in [random.choice([True, False])]:
            # different imputers
            for imp in [random.choice([KNNImputer(), SimpleImputer(strategy='mean'), SimpleImputer(strategy='most_frequent')])]:
                for scl in [random.choice([RobustScaler(), StandardScaler(), MinMaxScaler(), Normalizer()])]:
                    df_scores = baseline_models(
                        X_train, 
                        y_train, 
                        X_test, 
                        y_test, 
                        do_smote=b, 
                        algo_attempt_list=['bnb', 'gnb', 'lm'],
                        grid_search=True,
                        cv=5,
                        verbose=0,
                        imputer=imp,
                        scaler=scl
                    )
                    df_scores['SMOTE'] = b
                    df_scores['imputer'] = imp
                    df_scores['scaler'] = scl
                    # sort to keep best
                    df_scores.sort_values(
                        by=[primary_eval_metric, secondary_eval_metric], 
                        inplace=True, 
                        ascending=False
                    )
                    df_scores = df_scores.reset_index()
                    df_scores = df_scores.iloc[:,1:]
                    if df_scores[eval_metric][0] > best_score:
                        best_f1 = df_scores[eval_metric][0]
                        best_cols = df_random_feats.columns
                        df_scores['features'] = str(df_random_feats.columns.tolist())
                        df_scores['trial_num'] = i
                        df_best_scores = pd.concat([df_best_scores, df_scores.iloc[0:1,:]], ignore_index=True)
except KeyboardInterrupt as e:
    print(e)

df_best_scores.sort_values(
    by=[primary_eval_metric, secondary_eval_metric], 
    inplace=True, 
    ascending=False
)
df_best_scores = df_best_scores.reset_index(drop=True)
print(best_f1)
print(best_cols)
df_best_scores


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


print(df_best_scores.iloc[0,:]['features'])
X = df_cdc_clean[ast.literal_eval(df_best_scores.iloc[0,:]['features'])]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



baseline_models(
    X_train, 
    y_train, 
    X_test, 
    y_test, 
    do_smote=True,
    show_confusion_matrix=True,
    algo_attempt_list=['lm'],
    show_full_report=True,
    scaler=MinMaxScaler()	
    )


algo_name = 'Logistic Regression'
print(df_best_scores.iloc[0,:]['features'])
X = df_cdc_clean[keep_feats]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

my_imputer = SimpleImputer()
X_train = my_imputer.fit_transform(X_train)
X_test = my_imputer.fit_transform(X_test)

sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)


processing_pipeline = make_pipeline(SimpleImputer(), MinMaxScaler(), LogisticRegression(max_iter=1000, penalty='l2', C=10))


processing_pipeline.fit(X_train, y_train)
pred_labels  = processing_pipeline.predict(X_test)
pred_labels = [x.round() for x in pred_labels]

df_algo_cls_rpt = get_classification_report(pred_labels, y_test, algo_name, show_full_report=True)
tn, fp, fn, tp = confusion_matrix(pred_labels, y_test).ravel()
df_algo_cls_rpt['tp'] = tp
df_algo_cls_rpt['fn'] = fn
df_algo_cls_rpt['fp'] = fp
df_algo_cls_rpt['tn'] = tn
df_algo_cls_rpt


# ## With subset of selected features

feature_subset = [
    'age_liver_condition',
    'age_range_first_menstrual_period',
    'annual_healthcare_visit_count',
    'have_liver_condition',
    'type_of_work_done_last_week',
    'weight_change_intentional',
    'days_nicotine_substitute_used',
    'pain_relief_from_cardio_recoverytime',
    'feeling_down_depressed_hopeless',
    'feeling_tired_or_having_little_energy',
    'trouble_concentrating',
    'has_smoked_tabacco_last_5days',
    'alcoholic_drinks_past_12mo',
    'count_lost_10plus_pounds',
    'has_tried_to_lose_weight_12mo',
    'duration_last_healthcare_visit',
    'count_minutes_moderate_recreational_activity',
    'count_minutes_moderate_sedentary_activity',
    'has_overweight_diagnosis',
    'food_security_level_adult',
    'monthly_poverty_index_category',
    'monthly_poverty_index',
    'count_hours_worked_last_week',
    'education_level',
    'is_usa_born',
    'has_health_insurance',
    'has_health_insurance_gap'
 ]


import warnings
warnings.filterwarnings("ignore")


best_score = 0
primary_eval_metric = 'recall'
secondary_eval_metric = 'f1-score'

best_cols = []
df_best_scores = pd.DataFrame()

try:
    X = df_cdc_clean[feature_subset].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # SMOTE or not
    for b in [True, False]:
        # different imputers
        for imp in [KNNImputer(), SimpleImputer(strategy='mean'), SimpleImputer(strategy='most_frequent')]:
            for scl in [RobustScaler(), StandardScaler(), MinMaxScaler(), Normalizer()]:
                df_scores = baseline_models(
                    X_train, 
                    y_train, 
                    X_test, 
                    y_test, 
                    do_smote=b, 
                    algo_attempt_list='all',
                    grid_search=True,
                    cv=5,
                    verbose=0,
                    imputer=imp,
                    scaler=scl
                )
                # add setting from best trial to the output df
                df_scores['SMOTE'] = b
                df_scores['imputer'] = imp
                df_scores['scaler'] = scl
                if df_scores[eval_metric][0] > best_f1:
                    best_f1 = df_scores[eval_metric][0]
                    df_best_scores = df_scores.copy()
except KeyboardInterrupt as e:
    print(e)

print(best_f1)
print(feature_subset)
df_best_scores


# 

# 
