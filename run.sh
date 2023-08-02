#!/bin/sh
# set -x
IMAGE_NAME="621057158777.dkr.ecr.us-east-1.amazonaws.com/w210jhand"
IMAGE_TAG="latest"
CONTAINER_NAME="621057158777.dkr.ecr.us-east-1.amazonaws.com/w210jhand"
NAMESPACE="w210jhand"

echo "starting redis"
docker run -d --name temp-redis -p 6379:6379 redis
echo "redis started"

cd backend

# Run pytest within poetry virtualenv
poetry env remove python3.10
poetry install

# train model
echo "training model..."
poetry run python trainer/train.py

# move model to src
mv model_pipeline.pkl src/model_pipeline.pkl

# run tests
# poetry run pytest -vv -s

# minikube delete
minikube start --kubernetes-version=v1.25.4

# can build inside minikube 
eval $(minikube docker-env)
docker build -t $IMAGE_NAME:$IMAGE_TAG .
echo "container built..."

# # or can load existing image into minikube
# # saves build time if image is not changing
# echo "pushing $IMAGE_NAME:$IMAGE_TAG"
# minikube image load $IMAGE_NAME:$IMAGE_TAG
# echo "pushed existing container to minikube"

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
echo "health check"
curl -o /dev/null -s -w "%{http_code}\n" -X GET "http://localhost:8000/health"

echo "\ntest list of inputs:"

curl -X POST -H 'Content-Type: application/json' localhost:8000/predict -d \
'''
    {
        "surveys" : [
            {
            "little_interest_in_doing_things": 1.0,
            "feeling_down_depressed_hopeless": 1.0,
            "trouble_falling_or_staying_asleep": 0.0,
            "feeling_tired_or_having_little_energy": 0.0,
            "poor_appetitie_or_overeating": 0.0,
            "feeling_bad_about_yourself": 0.0,
            "trouble_concentrating": 0.0,
            "moving_or_speaking_to_slowly_or_fast": 0.0,
            "thoughts_you_would_be_better_off_dead": 0.0,
            "difficult_doing_daytoday_tasks": 0.0,
            "seen_mental_health_professional": 10.0,
            "times_with_12plus_alc": 11.0,
            "time_since_last_healthcare": 12.0,
            "cholesterol_prescription": 1.0,
            "high_cholesterol": 1.0,
            "age_in_years": 1.0,
            "horomones_not_bc": 16.0,
            "months_since_birth": 1.0,
            "arthritis": 18.0,
            "high_bp": 1.0,
            "regular_periods": 1.0,
            "moderate_recreation": 1.0,
            "thyroid_issues": 1.0,
            "vigorous_recreation": 1.0,
            "stroke": 1.0,
            "is_usa_born": 25.0,
            "asthma": 1.0,
            "count_days_moderate_recreational_activity": 1.0,
            "have_health_insurance": 10.0,
            "weight_lbs": 150.0,
            "height_in": 65.0,
            "count_lost_10plus_pounds": 31.0,
            "times_with_8plus_alc": 32.0,
            "duration_last_healthcare_visit": 33.0,
            "work_schedule": 34.0
            }
        ]
    }
'''

curl -X POST -H 'Content-Type: application/json' http://network-load-balancer-3ec3c60f32bd38c8.elb.us-west-1.amazonaws.com/predict -d \
'''
    {
        "surveys" : [
            {
            "little_interest_in_doing_things": 1.0,
            "feeling_down_depressed_hopeless": 1.0,
            "trouble_falling_or_staying_asleep": 0.0,
            "feeling_tired_or_having_little_energy": 0.0,
            "poor_appetitie_or_overeating": 0.0,
            "feeling_bad_about_yourself": 0.0,
            "trouble_concentrating": 0.0,
            "moving_or_speaking_to_slowly_or_fast": 0.0,
            "thoughts_you_would_be_better_off_dead": 0.0,
            "difficult_doing_daytoday_tasks": 0.0,
            "seen_mental_health_professional": 10.0,
            "times_with_12plus_alc": 11.0,
            "time_since_last_healthcare": 12.0,
            "cholesterol_prescription": 1.0,
            "high_cholesterol": 1.0,
            "age_in_years": 1.0,
            "horomones_not_bc": 16.0,
            "months_since_birth": 1.0,
            "arthritis": 18.0,
            "high_bp": 1.0,
            "regular_periods": 1.0,
            "moderate_recreation": 1.0,
            "thyroid_issues": 1.0,
            "vigorous_recreation": 1.0,
            "stroke": 1.0,
            "is_usa_born": 25.0,
            "asthma": 1.0,
            "count_days_moderate_recreational_activity": 1.0,
            "have_health_insurance": 10.0,
            "weight_lbs": 150.0,
            "height_in": 65.0,
            "count_lost_10plus_pounds": 31.0,
            "times_with_8plus_alc": 32.0,
            "duration_last_healthcare_visit": 33.0,
            "work_schedule": 34.0
            }
        ]
    }
'''



read -p "Press enter to continue"

kill $(pgrep kubectl)

kubectl delete --all service --namespace=$NAMESPACE
kubectl delete --all deployment --namespace=$NAMESPACE

kubectl delete namespace $NAMESPACE
minikube stop


export IMAGE_NAME="public.ecr.aws/f8e0d4w0/w210jhand"
export IMAGE_TAG="latest"
export NAMESPACE="w210jhand"