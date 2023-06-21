#!/bin/sh
# set -x
IMAGE_NAME="w210backend"
IMAGE_TAG="1.0.0"
CONTAINER_NAME="w210backendcontainer"
NAMESPACE="w210_jhand"

cd backend

# Run pytest within poetry virtualenv
poetry env remove python3.10
poetry install

# train model
poetry run python backend/trainer/train.py

# move model to src
mv backend/trainer/model_pipeline.pkl backend/src/model_pipeline.pkl

# run tests
poetry run pytest -vv -s

minikube delete
minikube start --kubernetes-version=v1.25.4
eval $(minikube docker-env)

docker build -t $IMAGE_NAME:$IMAGE_TAG .
echo "container built..."

cd infra

kubectl create -f ./namespace.yaml
kubectl apply -f deployment-redis.yaml -n $NAMESPACE
kubectl apply -f service-redis.yaml -n $NAMESPACE
kubectl apply -f deployment-pythonapi.yaml -n $NAMESPACE
kubectl apply -f service-prediction.yaml -n $NAMESPACE

# make sure k8 is ready
kubectl rollout status deployment pythonapi-deployment -n $NAMESPACE

kubectl port-forward -n $NAMESPACE service/prediction-service 8000:8000 &

# wait for the /health endpoint to return a 200 and then move on
finished=false
while ! $finished; do
    health_status=$(curl -o /dev/null -s -w "%{http_code}\n" -X GET "http://localhost:8000/health")
    if [ $health_status == "200" ]; then
        finished=true
        echo "API is ready"
    else
        echo "API not responding yet"
        sleep 1
    fi
done

echo "\ntest list of inputs:"
curl -X POST -H 'Content-Type: application/json' localhost:8000/predict -d \
'''
    {
        "surveys" : [
               "has_health_insurance": 1.0,
                "difficult_doing_daytoday_tasks": 0.0,
                "age_range_first_menstrual_period": 0,
                "weight_change_intentional": 0,
                "thoughts_you_would_be_better_off_dead": 0.0,
                "little_interest_in_doing_things": 1.0,
                "trouble_concentrating": 0.0,
                "food_security_level_household": 2.0,
                "general_health_condition": 4.0,
                "monthly_poverty_index": 2.0,
                "food_security_level_adult": 2.0,
                "count_days_seen_doctor_12mo": 4.0,
                "has_overweight_diagnosis": 1.0,
                "feeling_down_depressed_hopeless": 0.0,
                "count_minutes_moderate_recreational_activity": 15.0,
                "have_liver_condition": 0,
                "pain_relief_from_cardio_recoverytime": 1.0,
                "education_level": 5.0,
                "count_hours_worked_last_week": 40.0,
                "age_in_years": 44.0,
                "has_diabetes": 1.0,
                "alcoholic_drinks_past_12mo": 5.0,
                "count_lost_10plus_pounds": 3.0,
                "days_nicotine_substitute_used": 0,
                "age_with_angina_pectoris": 33.0,
                "annual_healthcare_visit_count": 3.0,
                "poor_appetitie_or_overeating": 1.0,
                "feeling_bad_about_yourself": 0.0,
                "has_tried_to_lose_weight_12mo": 0.0,
                "count_days_moderate_recreational_activity": 2.0,
                "count_minutes_moderate_sedentary_activity": 960.0
        ]
    }
'''

kill $(pgrep kubectl)

kubectl delete --all service --namespace=$NAMESPACE
kubectl delete --all deployment --namespace=$NAMESPACE

kubectl delete namespace $NAMESPACE
minikube stop