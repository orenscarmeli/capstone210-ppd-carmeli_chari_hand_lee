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
    'age_in_years': 32.0,
    'height_in': 62.0, 
    'weight_lbs': 150.0, 
    'is_usa_born': 1.0, 
    'have_health_insurance': 1.0, 
    'months_since_birth': 3.0, 
    'regular_periods': 1.0,
    'horomones_not_bc': 1.0, 
    'time_since_last_healthcare': 4.0,
    'seen_mental_health_professional': 1.0, 
    'little_interest_in_doing_things': 3.0, 
    'feeling_down_depressed_hopeless': 3.0, 
    'trouble_falling_or_staying_asleep': 3.0,
    'feeling_tired_or_having_little_energy': 3.0,
    'poor_appetitie_or_overeating': 3.0,
    'feeling_bad_about_yourself': 3.0, 
    'trouble_concentrating': 3.0,
    'moving_or_speaking_to_slowly_or_fast': 3.0,
    'thoughts_you_would_be_better_off_dead': 3.0, 
    'difficult_doing_daytoday_tasks': 3.0, 
    'times_with_12plus_alc': 0.0,
    'high_cholesterol': 1.0, 
    'cholesterol_prescription': 1.0, 
    'high_bp': 1.0, 
    'moderate_recreation': 1.0,
    'count_days_moderate_recreational_activity': 1.0,
    'vigorous_recreation': 1.0, 
    'thyroid_issues': 1.0, 
    'arthritis': 1.0,
    'stroke': 1.0,
    'asthma': 1.0,
    'count_lost_10plus_pounds': 4.0,
    'times_with_8plus_alc': 0.0, 
    'duration_last_healthcare_visit': 3.0, 
    'work_schedule': 5.0
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
        logging.warning(f"low model: {X_low}")
        pred = clf_low.predict(X_low)

    else:
        # num_dep_screener_0
        X_1 = np.append(X[:, :29], [[(X[0:, 0:11] == 0).sum()]], axis=1)
        # weight_lbs_over_height_in_ratio
        X_2 = np.append(X_1, [[(X[0, 29] / X[0, 30])]], axis=1)
        # impute and scale
        imputer_high = SimpleImputer(strategy="median")
        trans_high = RobustScaler()
        # X_high = imputer_high.fit_transform(X_2)
        # X_high = trans_high.fit_transform(X_high)
        logging.warning(f"high model: {X_high}")
        X_high = np.nan_to_num(X_high)
        pred = clf_high.predict(X_high)

    # pred = clf.predict(survey_features)
    pred = np.reshape(pred, (-1, 1))
    pred_formatted = [Prediction(prediction=p) for p in pred]
    preds = Predictions(predictions=pred_formatted)
    return preds