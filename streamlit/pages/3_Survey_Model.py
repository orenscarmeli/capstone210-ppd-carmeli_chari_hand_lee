import streamlit as st
from operator import mod
from os import getcwd
from os.path import exists, join

import joblib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression, LinearRegression
import warnings
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import  GradientBoostingClassifier
# import xgboost as xgb
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC, LinearSVC 
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import recall_score

from sklearn import tree
from sklearn.decomposition import PCA, SparsePCA

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import json
import pickle
import warnings
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from collections import Counter
import pydeck as pdk
import smtplib
from email.message import EmailMessage



st.set_page_config(page_title="Survey Model",
                   page_icon="./page_logo.png")

st.image("./header_survey.png")


st.sidebar.header("Survey Model")
st.write(
    """This generator asks questions predetermined to be relevant for predicting depression and uses responses in our trained model to predict depression"""
)


# Function to preprocess data and generate test/train split
def get_model_data(original_df, columns, test_size_prop=0.2):
    """
    Function to build feature & indicator matrices for both train & test.
    """
    
    # add target column (MDD)
    cols_to_use = columns.copy()
    cols_to_use.insert(0, 'MDD')
    
    df_to_use = original_df[cols_to_use]
    
    # Create test & train data
    x = df_to_use.iloc[:,1:].values
    y = df_to_use['MDD'].values
    
    # SimpleImputer() = fill in missing values
    # note imputer may drop columns if no values exist for it
    imputer = SimpleImputer(strategy='median')  
    x = imputer.fit_transform(x)

    # RobustScaler() = scale features to remove outliers
    trans = RobustScaler()
    x = trans.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(
        x, 
        y, 
        test_size=test_size_prop, 
        random_state=42
    ) 
    return x_train, x_test, y_train, y_test

def create_model():
    cdc_survey = pd.read_csv('/users/leej136/Downloads/cdc_nhanes_survey_responses_clean.csv')
    all_columns = [
    # Depression screener
    'little_interest_in_doing_things',
    'feeling_down_depressed_hopeless',
    'trouble_falling_or_staying_asleep',
    'feeling_tired_or_having_little_energy',
    'poor_appetitie_or_overeating',
    'feeling_bad_about_yourself',
    'trouble_concentrating',
    'moving_or_speaking_to_slowly_or_fast',
    'thoughts_you_would_be_better_off_dead',
    'difficult_doing_daytoday_tasks',
    # Alcohol & smoking
    'has_smoked_tabacco_last_5days',
    'alcoholic_drinks_past_12mo', 
    'drank_alc',
    'alc_drinking_freq',
    'alc_per_day',
    'times_with_4or5_alc',
    'times_with_8plus_alc',
    'times_with_12plus_alc',
    '4plus_alc_daily',
    'days_4plus_drinks_occasion',
    #Blood Pressure & Cholesterol
    'high_bp',
    'age_hypertension',
    'hypertension_prescription',
    'high_bp_prescription',
    'high_cholesterol',
    'cholesterol_prescription',
    #Cardiovascular Health
    'chest_discomfort',
    # Diet & Nutrition
    'how_healthy_is_your_diet',    
    'count_lost_10plus_pounds',
    'has_tried_to_lose_weight_12mo', 
    'breastfed',
    'milk_consumption_freq',
    'govmnt_meal_delivery',
    'nonhomemade_meals',
    'fastfood_meals',
    'readytoeat_meals',
    'frozen_pizza',
    #Food Security
    'emergency_food_received',
    'food_stamps_used',
    'wic_benefit_used',
    #Hospital Utilization & Access to Care
    'general_health',
    'regular_healthcare_place',
    'time_since_last_healthcare',
    'overnight_in_hospital',
    'seen_mental_health_professional',
    #Health Insurance
    'have_health_insurance',
    'have_private_insurance',
    'plan_cover_prescriptions',
    #Income
    'family_poverty_level',
    'family_poverty_level_category',
    #Medical Conditions
    'asthma',
    'anemia_treatment',
    'blood_transfusion',
    'arthritis',
    'heart_failure',
    'coronary_heart_disease',
    'angina_pectoris',
    'heart_attack',
    'stroke',
    'thyroid_issues',
    'respiratory_issues',
    'abdominal_pain',
    'gallstones',
    'gallbladder_surgery',
    'cancer',
    'dr_recommend_lose_weight',
    'dr_recommend_exercise',
    'dr_recommend_reduce_salt',
    'dr_recommend_reduce_fat',
    'currently_losing_weight',
    'currently_increase_exercise',
    'currently_reducing_salt',
    'currently_reducing_fat',
    'metal_objects',
    #Occupation
    'hours_worked',
    'over_35_hrs_worked',
    'work_schedule',
    #Physical Activity
    'vigorous_work',
    'walk_or_bicycle',
    'vigorous_recreation',
    'moderate_recreation',
    # Physical health & Medical History
    'count_days_seen_doctor_12mo',
    'duration_last_healthcare_visit',        
    'count_days_moderate_recreational_activity',   
    'count_minutes_moderate_recreational_activity',
    'count_minutes_moderate_sedentary_activity',
    'general_health_condition',    
    'has_diabetes',
    'has_overweight_diagnosis',  
    #Reproductive Health
    'regular_periods',
    'age_last_period',
    'try_pregnancy_1yr',
    'see_dr_fertility',
    'pelvic_infection',
    'pregnant_now',
    'pregnancy_count',
    'diabetes_pregnancy',
    'delivery_count',
    'live_birth_count',
    'age_at_first_birth',
    'age_at_last_birth',
    'months_since_birth',
    'horomones_not_bc',
    #Smoking
    'smoked_100_cigs',
    'currently_smoke',
    #Weight History
    'height_in',
    'weight_lbs',
    'attempt_weight_loss_1yr',
    # Demographic data
    'food_security_level_household',   
    'food_security_level_adult',    
    'monthly_poverty_index_category',
    'monthly_poverty_index',
    'count_hours_worked_last_week',
    'age_in_years',   
    'education_level',
    'is_usa_born',    
    'has_health_insurance',
    'has_health_insurance_gap'   
]
    x_train, x_test, y_train, y_test = get_model_data(cdc_survey, all_columns[0:3])
    model = LogisticRegression()
    model.fit(x_train, y_train)
    return model

# def send_email(subject, body):
#     """
#     Forwards survey prediction and answers to Velvaere email.
#     """
#     email_address = 'contact.velvaere@gmail.com'
#     email_password = 'mids2023!!'
    
#     msg = EmailMessage()
#     msg.set_content(body)

#     msg['Subject'] = subject
#     msg['From'] = email_address
#     msg['To'] = email_address

#     try:
#         # Connect to the SMTP server and send the email
#         server = smtplib.SMTP('smtp.gmail.com', 587)
#         server.starttls()
#         server.login(email_address, email_password)
#         server.send_message(msg)
#         server.quit()
#         return True
#     except Exception as e:
#         print(f"Failed to send email: {e}")
#         return False

# Main function
def main():
    model = create_model()

    # st.title("Postpartum Depression Screening")
    # st.write("Answer the following questions to predict postpartum depression")

    # Define the questions and answer options

    questions = [
        {
            "question": "Over the last 2 weeks, how often have you been bothered by the following problems: little interest or pleasure in doing things?",
            "options": ["Not at all", "Several days", "More than half the days", "Nearly every day"]
        },
        {
            "question": "Over the last 2 weeks, how often have you been bothered by feeling down, depressed, or hopeless?",
            "options": ["Not at all", "Several days", "More than half the days", "Nearly every day"]
        },
        {
            "question": "Over the last 2 weeks, how often have you been bothered by trouble falling or staying asleep, or sleeping too much?",
            "options": ["Not at all", "Several days", "More than half the days", "Nearly every day"]
        }
    ]

    # Display the questions and collect answers in a form
    with st.form(key='survey_form'):
        answers = []
        # Add invisible dummy radio button to hide preselection
        st.markdown(
            """
        <style>
            div[role=radiogroup] label:first-of-type {
                visibility: hidden;
                height: 0px;
            }
        </style>
        """,
            unsafe_allow_html=True,
        )

        # Present questions with answer options
        for question in questions:
            answer = st.radio(question["question"], question["options"])
            answer_int = question["options"].index(answer)
            answers.append(answer_int)

        # Make prediction using the model
        feature_matrix = [answers]  # Convert answers to a feature matrix
        prediction = model.predict(feature_matrix)

        # Convert prediction label to description
        if prediction[0] == 1:
            prediction_text = " __% likely to have Postpartum Depression. We recommend reaching out to your healthcare provider for further evaluation."
        else:
            prediction_text = "__% likely to have Postpartum Depression. Our model predicts a low likelihood of PPD but please reach out to your healthcare provider if you feel the need for further evaluation."

        submit = st.form_submit_button(label='Submit')

    # Show the prediction after submission
    if submit:
        st.write("**Prediction:**", prediction_text)
    
    # # Provide user with turnaround window and confirmation of answer submission
    # if submit:
    #     # st.write("Thank you for your submission! A physician will review your answers and a member of our team will reach out within 2 business days.")
    #     # Send the prediction result and answers to the specified email address
    #     to_email = 'velvaere@gmail.com'
    #     subject = 'Postpartum Depression Prediction Results'
    #     body = f"Prediction: {prediction_text}\n\nAnswers: {answers}"
    #     email_sent = send_email(subject, body)

    #     if email_sent:
    #         st.write("Thank you for your submission! A physician will review your answers and a member of our team will reach out within 2 business days.")
    #     else:
    #         st.error("Failed to send the email.")



# Run the app
if __name__ == "__main__":
    main()