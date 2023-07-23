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
clf = joblib.load(os.path.join(dirname, "model_pipeline.pkl"))


# Use pydantic.Extra.forbid to only except exact field set from client.
class Survey(BaseModel, extra=Extra.forbid):
    little_interest_in_doing_things: float | None = None
    feeling_down_depressed_hopeless: float | None = None
    trouble_falling_or_staying_asleep: float | None = None
    feeling_tired_or_having_little_energy: float | None = None
    poor_appetitie_or_overeating: float | None = None
    feeling_bad_about_yourself: float | None = None
    trouble_concentrating: float | None = None
    moving_or_speaking_to_slowly_or_fast: float | None = None
    thoughts_you_would_be_better_off_dead: float | None = None
    difficult_doing_daytoday_tasks: float | None = None
    times_with_12plus_alc: float | None = None
    seen_mental_health_professional: float | None = None
    count_days_seen_doctor_12mo: float | None = None
    count_lost_10plus_pounds: float | None = None
    arthritis: float | None = None
    horomones_not_bc: float | None = None
    is_usa_born: float | None = None
    times_with_8plus_alc: float | None = None
    time_since_last_healthcare: float | None = None
    duration_last_healthcare_visit: float | None = None
    work_schedule: float | None = None
    age_in_years: float | None = None
    regular_periods: float | None = None
    count_minutes_moderate_sedentary_activity: float | None = None
    emergency_food_received: float | None = None
    high_bp: float | None = None
    dr_recommend_exercise: float | None = None
    metal_objects: float | None = None
    drank_alc: float | None = None
    cholesterol_prescription: float | None = None
    smoked_100_cigs: float | None = None
    vigorous_recreation: float | None = None
    dr_recommend_lose_weight: float | None = None
    cancer: float | None = None
    chest_discomfort: float | None = None
    has_health_insurance: float | None = None
    have_health_insurance: float | None = None
    weight_lbs: float | None = None
    readytoeat_meals: float | None = None
    regular_healthcare_place: float | None = None
    try_pregnancy_1yr: float | None = None
    currently_increase_exercise: float | None = None
    coronary_heart_disease: float | None = None
    stroke: float | None = None
    heart_attack: float | None = None
    see_dr_fertility: float | None = None


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
    redis_url = "redis-service"
    redis = aioredis.from_url(
        f"redis://{redis_url}:6379", encoding="utf8", decode_responses=True
    )
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")


@app.get("/")
async def root():
    current_time = datetime.now().isoformat()
    return {"message": current_time}


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
    survey_features = np.array(list(survey_list))
    pred = clf.predict(survey_features)
    pred = np.reshape(pred, (-1, 1))
    pred_formatted = [Prediction(prediction=p) for p in pred]
    preds = Predictions(predictions=pred_formatted)
    return preds
