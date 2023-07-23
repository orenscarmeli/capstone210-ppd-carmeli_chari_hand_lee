#!/bin/sh
# set -x
IMAGE_NAME="public.ecr.aws/f8e0d4w0/w210jhand"
CONTAINER_NAME="lab4container"
# had to add sed at the end to take period out between jon.hand
NAMESPACE="w210jhand"
#621057158777.dkr.ecr.us-east-1.amazonaws.com/w210jhand
# short id of most recent commit
TAG=$(git log -n1 --format="%h")
AWS_DEFAULT_REGION="us-west-1"
AWS_ACCESS_KEY_ID="AKIAZBGPEFZ42LQHR36S"
# FQDN = Fully-Qualified Domain Name

# IMAGE_FQDN="${ACR_DOMAIN}/${NAMESPACE}/${IMAGE_NAME}"
# echo $IMAGE_FQDN

cd backend


# az login --tenant berkeleydatasciw255.onmicrosoft.com
# az acr login --name w255mids

# ########## dev ##############
# minikube delete
# minikube start --kubernetes-version=v1.25.4
# kubectl config use-context minikube -n $NAMESPACE
# eval $(minikube docker-env)
# docker build -t $IMAGE_NAME:$TAG .
# echo "container built..."
# kubectl apply -k .k8s/overlays/dev -n $NAMESPACE

########## prod ##############

###### Azure Specirfic ######
# kubectl config use-context w255-aks -n $NAMESPACE
# docker build --platform linux/amd64 -t $ACR_DOMAIN/$NAMESPACE/$IMAGE_NAME:$TAG .
# # docker tag ${IMAGE_NAME} ${IMAGE_FQDN}

# docker push $ACR_DOMAIN/$NAMESPACE/$IMAGE_NAME:$TAG
# docker pull $ACR_DOMAIN/$NAMESPACE/$IMAGE_NAME:$TAG
# echo "container built..."

###### AWS ECR Specific ######
# kubectl config use-context jon.hand@test-cluster.us-west-1.eksctl.io -n $NAMESPACE
# docker buildx build --platform linux/amxd64 -t $IMAGE_NAME:$TAG .

aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws/f8e0d4w0
docker buildx build -t w210jhand:$TAG --platform=linux/amd64 .

docker tag w210jhand:$TAG public.ecr.aws/f8e0d4w0/w210jhand:$TAG

docker push public.ecr.aws/f8e0d4w0/w210jhand:$TAG


# # EKS
# kubectl create -f ./namespace.yaml
# kubectl apply -f deployment-redis.yaml -n $NAMESPACE
# kubectl apply -f service-redis.yaml -n $NAMESPACE
# kubectl apply -f deployment-pythonapi.yaml -n $NAMESPACE
# kubectl apply -f service-prediction.yaml -n $NAMESPACE

# # sub tag with the latest commit
# sed "s/\[TAG\]/${TAG}/g" .k8s/overlays/prod/patch-deployment-lab4_copy.yaml > .k8s/overlays/prod/patch-deployment-lab4.yaml

# make sure we don't have other virtual services running
# kubectl delete --all virtualservice --namespace=$NAMESPACE

# apply overlays
# kubectl apply -k .k8s/overlays/prod -n $NAMESPACE

# make sure k8 is ready
# kubectl rollout status deployment pythonapi-deployment -n $NAMESPACE

# test predict endpoint
curl -X POST -H 'Content-Type: application/json' 'http://network-load-balancer-3ec3c60f32bd38c8.elb.us-west-1.amazonaws.com/predict' -d \
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
            "difficult_doing_daytoday_tasks": 1.0,
            "times_with_12plus_alc": 0.0,
            "seen_mental_health_professional": 2.0,
            "count_days_seen_doctor_12mo": 1.0,
            "count_lost_10plus_pounds": 0.0,
            "arthritis": 1.0,
            "horomones_not_bc": 2.0,
            "is_usa_born": 1.0,
            "times_with_8plus_alc": 0.0,
            "time_since_last_healthcare": 0.0,
            "duration_last_healthcare_visit": 0.0,
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
            "try_pregnancy_1yr": 0.0,
            "currently_increase_exercise": 1.0,
            "coronary_heart_disease": 2.0,
            "stroke": 2.0,
            "heart_attack": 2.0,
            "see_dr_fertility": 0.0
            }
        ]
    }
'''
# with some missing data
curl -X POST -H 'Content-Type: application/json' 'http://network-load-balancer-3ec3c60f32bd38c8.elb.us-west-1.amazonaws.com/predict' -d \
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
            "difficult_doing_daytoday_tasks": 1.0,
            "seen_mental_health_professional": 2.0,
            "count_days_seen_doctor_12mo": 1.0,
            "arthritis": 1.0,
            "horomones_not_bc": 2.0,
            "is_usa_born": 1.0,
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
            "currently_increase_exercise": 1.0,
            "coronary_heart_disease": 2.0,
            "stroke": 2.0,
            "heart_attack": 2.0,
            }
        ]
    }
'''
# curl -o /dev/null -s -w "%{http_code}\n" -X GET 'http://network-load-balancer-867733373408bca0.elb.us-west-1.amazonaws.com/health'


# kubectl delete --all service --namespace=$NAMESPACE
# kubectl delete --all deployment --namespace=$NAMESPACE

