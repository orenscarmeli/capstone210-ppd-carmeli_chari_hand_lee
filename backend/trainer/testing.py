import joblib
import logging
import numpy as np
import pandas as pd
import os
import logging
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, KBinsDiscretizer


dep_screener_cols = [
    "little_interest_in_doing_things",
    "feeling_down_depressed_hopeless",
    "trouble_falling_or_staying_asleep",
    "feeling_tired_or_having_little_energy",
    "poor_appetitie_or_overeating",
    "feeling_bad_about_yourself",
    "trouble_concentrating",
    "moving_or_speaking_to_slowly_or_fast",
    "thoughts_you_would_be_better_off_dead",
    "difficult_doing_daytoday_tasks",
]

dirname = os.path.dirname(__file__)
clf_low = joblib.load(os.path.join(dirname, "model_pipeline_low.pkl"))
clf_high = joblib.load(os.path.join(dirname, "model_pipeline_high.pkl"))
data = {
    "little_interest_in_doing_things": 1,
    "feeling_down_depressed_hopeless": 1,
    "trouble_falling_or_staying_asleep": 0,
    "feeling_tired_or_having_little_energy": 0,
    "poor_appetitie_or_overeating": 0,
    "feeling_bad_about_yourself": 0,
    "trouble_concentrating": 0,
    "moving_or_speaking_to_slowly_or_fast": 0,
    "thoughts_you_would_be_better_off_dead": 0,
    "difficult_doing_daytoday_tasks": 0,
    "seen_mental_health_professional": 10,
    "times_with_12plus_alc": 11,
    "time_since_last_healthcare": 12,
    "cholesterol_prescription": 1,
    "high_cholesterol": 1,
    "age_in_years": 1,
    "horomones_not_bc": 16,
    "months_since_birth": 1,
    "arthritis": 18,
    "high_bp": 1,
    "regular_periods": 1,
    "moderate_recreation": 1,
    "thyroid_issues": 1,
    "vigorous_recreation": 1,
    "stroke": 1,
    "is_usa_born": 25,
    "asthma": 1,
    "count_days_moderate_recreational_activity": 1,
    "have_health_insurance": 10,
    "weight_lbs": 150,
    "height_in": 65,
    ### need to do preprocessing to create these
    # "num_dep_screener_0": 1,
    # "weight_lbs_over_height_in_ratio": 1,
    # 'count_days_seen_doctor_12mo_bin',
    ### low columns
    # 'times_with_12plus_alc':11,
    # 'seen_mental_health_professional':10,
    "count_lost_10plus_pounds": 31,
    # 'arthritis':19,
    # 'horomones_not_bc':16,
    # 'is_usa_born':24,
    "times_with_8plus_alc": 32,
    # 'time_since_last_healthcare':12,
    "duration_last_healthcare_visit": 33,
    "work_schedule": 34,
}


X = np.array([[np.nan if val is None else val for val in data.values()]])
# use low
if (X[0:, 0:11] == 0).sum() >= 9:
    # subset to features for this model
    X_low = np.take(X, [11, 10, 31, 18, 16, 25, 32, 12, 33, 34], axis=1)
    if not np.isnan(X_low[0, 1]):
        # count_days_seen_doctor_12mo_bin
        # create bins using estimator
        est = KBinsDiscretizer(
            n_bins=10, encode="ordinal", strategy="uniform", subsample=None
        )

        feature_values = np.array([X_low[0, 1]]).reshape([-1, 1])
        est.fit(feature_values)
        feature_values = est.transform(feature_values)
        X_low = np.append([[np.nan]], X_low, axis=1)
    else:
        X_low = np.append([[np.nan]], X_low, axis=1)
    # impute and scale
    imputer_low = SimpleImputer(strategy="median")
    trans_low = RobustScaler()
    # X_low = imputer_low.fit_transform(X_low)
    # X_low = trans_low.fit_transform(X_low)
    X_low = np.nan_to_num(X_low)

    pred = clf_low.predict(X_low)

else:
    # num_dep_screener_0
    X_1 = np.append(X[:, :29], [[(X[0:, 0:11] == 0).sum()]], axis=1)
    # weight_lbs_over_height_in_ratio
    X_2 = np.append(X_1, [[(X[0, 29] / X[0, 30])]], axis=1)
    # impute and scale
    imputer_high = SimpleImputer(strategy="median")
    trans_high = RobustScaler()
    X_high = imputer_high.fit_transform(X_2)
    X_high = trans_high.fit_transform(X_high)

    pred = clf_high.predict(X_high)

print(pred)

# print(clf.predict(X))


# @app.post("/predict", response_model=Predictions)
# @cache(expire=60)
# async def predict(survey_input: surveys):
#     # logging.warning("in predict")
#     survey_list = [list(vars(s).values()) for h in survey_input.surveys]
#     survey_features = np.array(
#         list(survey_list)
#     )
#     pred = clf.predict(survey_features)
#     pred = np.reshape(pred, (-1, 1))
#     pred_formatted = [Prediction(prediction=p) for p in pred]
#     preds = Predictions(predictions=pred_formatted)
#     return preds
