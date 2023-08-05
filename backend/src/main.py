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


# Use pydantic.Extra.forbid to only except exact field set from client.
class Survey(BaseModel):  # , extra=Extra.forbid):
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
    seen_mental_health_professional: float | None = None
    times_with_12plus_alc: float | None = None
    time_since_last_healthcare: float | None = None
    cholesterol_prescription: float | None = None
    high_cholesterol: float | None = None
    age_in_years: float | None = None
    horomones_not_bc: float | None = None
    months_since_birth: float | None = None
    arthritis: float | None = None
    high_bp: float | None = None
    regular_periods: float | None = None
    moderate_recreation: float | None = None
    thyroid_issues: float | None = None
    vigorous_recreation: float | None = None
    stroke: float | None = None
    is_usa_born: float | None = None
    asthma: float | None = None
    count_days_moderate_recreational_activity: float | None = None
    have_health_insurance: float | None = None
    weight_lbs: float | None = None
    height_in: float | None = None
    count_lost_10plus_pounds: float | None = None
    times_with_8plus_alc: float | None = None
    duration_last_healthcare_visit: float | None = None
    work_schedule: float | None = None


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
    X = np.array([np.nan if val is None else val for val in survey_list])
    logging.warning(f"X: {X}")

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
        X = np.nan_to_num(X)
        # num_dep_screener_0
        X_1 = np.append(X[:, :29], [[(X[0:, 0:11] == 0).sum()]], axis=1)
        # weight_lbs_over_height_in_ratio
        if X[0, 30] == 0.0:
            X_2 = np.append(X_1, [[0.0]], axis=1)
        else:
            X_2 = np.append(X_1, [[(X[0, 29] / X[0, 30])]], axis=1)
        # impute and scale
        imputer_high = SimpleImputer(strategy="median")
        trans_high = RobustScaler()
        # X_high = imputer_high.fit_transform(X_2)
        # X_high = trans_high.fit_transform(X_high)
        X_high = X_2.copy()
        logging.warning(f"high model: {X_high}")
        
        X_high = np.nan_to_num(X_high)
        pred = clf_high.predict(X_high)

    # pred = clf.predict(survey_features)
    pred = np.reshape(pred, (-1, 1))
    pred_formatted = [Prediction(prediction=p) for p in pred]
    preds = Predictions(predictions=pred_formatted)
    return preds
