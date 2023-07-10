from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Extra, ValidationError, validator, parse_obj_as
from datetime import datetime

from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from redis import Redis
from redis import asyncio as aioredis

import joblib
import logging
import numpy as np
import os
import logging

dirname = os.path.dirname(__file__)
clf = joblib.load(os.path.join(dirname, 'model_pipeline.pkl'))

# Use pydantic.Extra.forbid to only except exact field set from client.
class Survey(BaseModel, extra=Extra.forbid):
    has_health_insurance: float | None = None
    difficult_doing_daytoday_tasks: float | None = None
    age_range_first_menstrual_period: float | None = None
    weight_change_intentional: float | None = None
    thoughts_you_would_be_better_off_dead: float | None = None
    little_interest_in_doing_things: float | None = None
    trouble_concentrating: float | None = None
    food_security_level_household: float | None = None
    general_health_condition: float | None = None
    monthly_poverty_index: float | None = None
    food_security_level_adult: float | None = None
    count_days_seen_doctor_12mo: float | None = None
    has_overweight_diagnosis: float | None = None
    feeling_down_depressed_hopeless: float | None = None
    count_minutes_moderate_recreational_activity: float | None = None
    have_liver_condition: float | None = None
    pain_relief_from_cardio_recoverytime: float | None = None
    education_level: float | None = None
    count_hours_worked_last_week: float | None = None
    age_in_years: float | None = None
    has_diabetes: float | None = None
    alcoholic_drinks_past_12mo: float | None = None
    count_lost_10plus_pounds: float | None = None
    days_nicotine_substitute_used: float | None = None
    age_with_angina_pectoris: float | None = None
    annual_healthcare_visit_count: float | None = None
    poor_appetitie_or_overeating: float | None = None
    feeling_bad_about_yourself: float | None = None
    has_tried_to_lose_weight_12mo: float | None = None
    count_days_moderate_recreational_activity: float | None = None
    count_minutes_moderate_sedentary_activity: float | None = None

    # to convert to numpy array
    def to_np(self):
        return np.array(list(vars(self).values())).reshape(1, -1)

    # # validators
    # @validator('MedInc')
    # def MedInc_must_be_positive(cls, v):
    #     if v <= 0:
    #         raise ValueError('must be positive')
    #     return v


class Surveys(BaseModel):
    surveys: list[Survey]

class Prediction(BaseModel):
    prediction: float

class Predictions(BaseModel):
    predictions: list[Prediction]

app = FastAPI()
logger = logging.getLogger("api")

@app.on_event("startup")
async def startup():
    # uses environment var in yaml
    # defaults to localhost if not found
    # redis_url = os.environ.get('REDIS_URL', 'localhost')
    # redis_url = os.environ.get('REDIS_URL', 'localhost')
    redis_url = 'redis-service'
    redis = aioredis.from_url(f"redis://{redis_url}:6379", encoding="utf8", decode_responses=True)
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")

@app.get("/")
async def root():
    current_time = datetime.now().isoformat()
    return {"message": current_time}

@app.get("/health")
async def get_health():
    current_time = datetime.now().isoformat()
    return {"message": current_time}


# @app.post("/predict", response_model=Predictions)
# @cache(expire=60)
@app.post("/predict", response_model=Predictions)
async def predict(survey_input: Surveys):
    # logging.warning("in predict")
    survey_list = [list(vars(s).values()) for s in survey_input.surveys]
    survey_features = np.array(
        list(survey_list)
    )
    pred = clf.predict(survey_features)
    pred = np.reshape(pred, (-1, 1))
    pred_formatted = [Prediction(prediction=p) for p in pred]
    preds = Predictions(predictions=pred_formatted)
    return preds

