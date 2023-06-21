
import joblib
import logging
import numpy as np
import os
import logging

dirname = os.path.dirname(__file__)
clf = joblib.load(os.path.join(dirname, 'model_pipeline.pkl'))
data = {
    'has_health_insurance': 1.0,
    'difficult_doing_daytoday_tasks': 0.0,
    'age_range_first_menstrual_period': 0,
    'weight_change_intentional': 0,
    'thoughts_you_would_be_better_off_dead': 0.0,
    'little_interest_in_doing_things': 1.0,
    'trouble_concentrating': 0.0,
    'food_security_level_household': 2.0,
    'general_health_condition': 4.0,
    'monthly_poverty_index': 2.0,
    'food_security_level_adult': 2.0,
    'count_days_seen_doctor_12mo': 4.0,
    'has_overweight_diagnosis': 1.0,
    'feeling_down_depressed_hopeless': 0.0,
    'count_minutes_moderate_recreational_activity': 15.0,
    'have_liver_condition': 0,
    'pain_relief_from_cardio_recoverytime': 1.0,
    'education_level': 5.0,
    'count_hours_worked_last_week': 40.0,
    'age_in_years': 44.0,
    'has_diabetes': 1.0,
    'alcoholic_drinks_past_12mo': 5.0,
    'count_lost_10plus_pounds': 3.0,
    'days_nicotine_substitute_used': 0,
    'age_with_angina_pectoris': 33.0,
    'annual_healthcare_visit_count': 3.0,
    'poor_appetitie_or_overeating': 1.0,
    'feeling_bad_about_yourself': 0.0,
    'has_tried_to_lose_weight_12mo': 0.0,
    'count_days_moderate_recreational_activity': 2.0,
    'count_minutes_moderate_sedentary_activity': 960.0
    }
X = np.array([list(data.values())])
# X = X.reshape(-1, 1)
print(X)
print(clf.predict(X))


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

