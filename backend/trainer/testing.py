import joblib
import logging
import numpy as np
import os
import logging

dirname = os.path.dirname(__file__)
clf = joblib.load(os.path.join(dirname, "model_pipeline.pkl"))
data = {
    "little_interest_in_doing_things": 1.0,
    "feeling_down_depressed_hopeless": 1.0,
    "trouble_falling_or_staying_asleep": 0.0,
    "feeling_tired_or_having_little_energy": 0.0,
    "poor_appetitie_or_overeating": 0.0,
    "feeling_bad_about_yourself": 0.0,
    "trouble_concentrating": 0.0,
    "moving_or_speaking_to_slowly_or_fast": 0.0,
    "thoughts_you_would_be_better_off_dead": 0.0,
    "difficult_doing_daytoday_tasks": 1.0,
    "times_with_12plus_alc": None,
    "seen_mental_health_professional": 2.0,
    "count_days_seen_doctor_12mo": 1.0,
    "count_lost_10plus_pounds": None,
    "arthritis": 1.0,
    "horomones_not_bc": 2.0,
    "is_usa_born": 1.0,
    "times_with_8plus_alc": None,
    "time_since_last_healthcare": None,
    "duration_last_healthcare_visit": None,
    "work_schedule": 2.0,
    "age_in_years": 68.0,
    "regular_periods": 2.0,
    "count_minutes_moderate_sedentary_activity": 180.0,
    "emergency_food_received": 2.0,
    "high_bp": 1.0,
    "dr_recommend_exercise": 1.0,
    "metal_objects": 2.0,
    "drank_alc": 1.0,
    "cholesterol_prescription": 1.0,
    "smoked_100_cigs": 2.0,
    "vigorous_recreation": 2.0,
    "dr_recommend_lose_weight": 2.0,
    "cancer": 2.0,
    "chest_discomfort": 2.0,
    "has_health_insurance": 1.0,
    "have_health_insurance": 1.0,
    "weight_lbs": 155.0,
    "readytoeat_meals": 3.0,
    "regular_healthcare_place": 1.0,
    "try_pregnancy_1yr": None,
    "currently_increase_exercise": 1.0,
    "coronary_heart_disease": 2.0,
    "stroke": 2.0,
    "heart_attack": 2.0,
    "see_dr_fertility": None,
}
X = np.array([[np.nan if val is None else val for val in data.values()]])
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
