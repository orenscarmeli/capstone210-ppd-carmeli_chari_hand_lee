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
    has_health_insurance: float
    difficult_doing_daytoday_tasks: float
    age_range_first_menstrual_period: float
    weight_change_intentional: float
    thoughts_you_would_be_better_off_dead: float
    little_interest_in_doing_things: float
    trouble_concentrating: float
    food_security_level_household: float
    general_health_condition: float
    monthly_poverty_index: float
    food_security_level_adult: float
    count_days_seen_doctor_12mo: float
    has_overweight_diagnosis: float
    feeling_down_depressed_hopeless: float
    count_minutes_moderate_recreational_activity: float
    have_liver_condition: float
    pain_relief_from_cardio_recoverytime: float
    education_level: float
    count_hours_worked_last_week: float
    age_in_years: float
    has_diabetes: float
    alcoholic_drinks_past_12mo: float
    count_lost_10plus_pounds: float
    days_nicotine_substitute_used: float
    age_with_angina_pectoris: float
    annual_healthcare_visit_count: float
    poor_appetitie_or_overeating: float
    feeling_bad_about_yourself: float
    has_tried_to_lose_weight_12mo: float
    count_days_moderate_recreational_activity: float
    count_minutes_moderate_sedentary_activity: float

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
    
    # @validator('prediction')
    # def prediction_must_be_positive(cls, v):
    #     if v <= 0:
    #         raise ValueError('must be positive')
    #     return v

class Predictions(BaseModel):
    predictions: list[Prediction]

app = FastAPI()
logger = logging.getLogger("api")

# caching
@cache(expire=60)
async def get_cache():
    return 1

@app.on_event("startup")
async def startup():
    # uses environment var in yaml
    # defaults to localhost if not found
    # redis_url = os.environ.get('REDIS_URL', 'localhost')
    redis_url = os.environ.get('REDIS_URL', 'localhost')
    redis = aioredis.from_url(f"redis://{redis_url}:6379", encoding="utf8", decode_responses=True)
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")
    
# status code 422 is for requests with semantic error
# hitting the endpoint with with an incorrect param 
# or no param will raise this error
# @app.get("/hello")
# async def read_hello(name: str):
#     return {"message": "hello " + name}

@app.get("/health")
async def get_health():
    
    current_time = datetime.now().isoformat()
    logging.warning(current_time)
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

