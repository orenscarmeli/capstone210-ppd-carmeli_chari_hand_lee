#!/bin/sh
# set -x
IMAGE_NAME="public.ecr.aws/f8e0d4w0/w210jhand"
CONTAINER_NAME="lab4container"
# had to add sed at the end to take period out between jon.hand
NAMESPACE="w210jhand"
#621057158777.dkr.ecr.us-east-1.amazonaws.com/w210jhand
# short id of most recent commit
TAG="latest"

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


# sub tag with the latest commit
sed "s/\[TAG\]/${TAG}/g" .k8s/overlays/prod/patch-deployment-lab4_copy.yaml > .k8s/overlays/prod/patch-deployment-lab4.yaml

##### AWS ECR ######
# docker buildx build --platform linux/amd64 -t $IMAGE_NAME:$TAG .

# make sure we don't have other virtual services running
# kubectl delete --all virtualservice --namespace=$NAMESPACE

# apply overlays
kubectl apply -k .k8s/overlays/prod -n $NAMESPACE

# make sure k8 is ready
kubectl rollout status deployment pythonapi-deployment -n $NAMESPACE

# test predict endpoint
# curl -X POST -H 'Content-Type: application/json' localhost:8000/predict -d \
# '''
#     {
#         "surveys" : [ 
#             {
#                "has_health_insurance": 1.0,
#                 "difficult_doing_daytoday_tasks": 0.0,
#                 "age_range_first_menstrual_period": 0,
#                 "weight_change_intentional": 0,
#                 "thoughts_you_would_be_better_off_dead": 0.0,
#                 "little_interest_in_doing_things": 1.0,
#                 "trouble_concentrating": 0.0,
#                 "food_security_level_household": 2.0,
#                 "general_health_condition": 4.0,
#                 "monthly_poverty_index": 2.0,
#                 "food_security_level_adult": 2.0,
#                 "count_days_seen_doctor_12mo": 4.0,
#                 "has_overweight_diagnosis": 1.0,
#                 "feeling_down_depressed_hopeless": 0.0,
#                 "count_minutes_moderate_recreational_activity": 15.0,
#                 "have_liver_condition": 0,
#                 "pain_relief_from_cardio_recoverytime": 1.0,
#                 "education_level": 5.0,
#                 "count_hours_worked_last_week": 40.0,
#                 "age_in_years": 44.0,
#                 "has_diabetes": 1.0,
#                 "alcoholic_drinks_past_12mo": 5.0,
#                 "count_lost_10plus_pounds": 3.0,
#                 "days_nicotine_substitute_used": 0,
#                 "age_with_angina_pectoris": 33.0,
#                 "annual_healthcare_visit_count": 3.0,
#                 "poor_appetitie_or_overeating": 1.0,
#                 "feeling_bad_about_yourself": 0.0,
#                 "has_tried_to_lose_weight_12mo": 0.0,
#                 "count_days_moderate_recreational_activity": 2.0,
#                 "count_minutes_moderate_sedentary_activity": 960.0
#             }
#         ]
#     }
# '''

# kubectl delete --all service --namespace=$NAMESPACE
# kubectl delete --all deployment --namespace=$NAMESPACE

