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

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """

st.set_page_config(page_title="Survey Model",
                   page_icon="./page_logo.png")

st.markdown(hide_default_format, unsafe_allow_html=True)

st.image("./header_survey.png")


st.sidebar.header("Survey Model")
st.write(
    """This generator asks questions predetermined to be relevant for predicting depression and uses responses in our trained model to predict depression."""
)

st.write("Please view our privacy policy for details regarding data prior to using the models.")

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



# Main function
def main():
    model = create_model()


    # Provide the questions and answer options
######  'num_dep_screener_0','weight_lbs_over_height_in_ratio' 

    questions = [
        {
            #RIDAGEYR: age_in_years
            "question": "How old are you? (Please select the closest option)",
            "options": list(range(0,86)),
            "var_code": "age_in_years"
        },
        {
            #height: height_in for weight_lbs_over_height_in_ratio
            "question": "What is your height in inches? (Please select the closest option)",
            "options": list(range(48,82)),
            "var_code": "height_in" 
        },
        {
            #weight: weight_lbs for weight_lbs_over_height_in_ratio
            "question": "What is your weight in pounds? (Please select the closest option)",
            "options": list(range(75,494)),
            "var_code": "weight_lbs"  
        },
        {
            #DMDBORN4: is_usa_born
            "question": "Were you born in the United States?",
            "options": ["Dummy","Yes","No"],
            "var_code": "is_usa_born"
        },
        {
            #HIQ011: have_health_insurance
            "question": "Are you covered by health insurance or some kind of health care plan?",
            "options": ["Dummy","Yes","No"],
            "var_code": "have_health_insurance"  
        },
        {
            #RHQ197: months_since_birth
            "question": "How many months ago did you have your baby? (Please select the closest option)",
            "options": list(range(1,28)),
            "var_code": "months_since_birth" 
        },
        {
            #RHQ031: regular_periods
            "question": "Have you had at least one menstrual period in the past 12 months?",
            "options": ["Dummy","Yes","No"],
            "var_code": "regular_periods" 
        },
        {
            #RHQ540: horomones_not_bc
            "question": "Have you ever used female hormones such as estrogen and progesterone? Please include any forms such as pills, cream, patch, and inejctables, but do not include birth controls methods or use for infertility.",
            "options": ["Dummy","Yes","No"],
            "var_code": "horomones_not_bc" 
        },
        {
            #HUD062: time_since_last_healthcare
            "question": "About how long has it been since you last saw a health care professional about your health for any reason?",
            "options": ["Dummy","Never","Within the past year","Within the last 2 years","Within the last 5 years","5 years ago or more"],
            "var_code": "time_since_last_healthcare" 
        },
        {
            #HUQ090: seen_mental_health_professional 
            "question": "During the past 12 months, have you seen or talked to a mental health professional about your health?",
            "options": ["Dummy","Yes","No"],
            "var_code": "seen_mental_health_professional" 
        },
        {
            #DPQ010: little_interest_in_doing_things 
            "question": "Over the last 2 weeks, how often have you been bothered by the following problems: little interest or pleasure in doing things?",
            "options": ["Dummy","Not at all", "Several days", "More than half the days", "Nearly every day"],
            "var_code": "little_interest_in_doing_things" 
        },
        {
            #DPQ020: feeling_down_depressed_hopeless 
            "question": "Over the last 2 weeks, how often have you been bothered by feeling down, depressed, or hopeless?",
            "options": ["Dummy","Not at all", "Several days", "More than half the days", "Nearly every day"],
            "var_code": "feeling_down_depressed_hopeless" 
        },
        {
            #DPQ030: trouble_falling_or_staying_asleep 
            "question": "Over the last 2 weeks, how often have you been bothered by trouble falling or staying asleep, or sleeping too much?",
            "options": ["Dummy","Not at all", "Several days", "More than half the days", "Nearly every day"],
            "var_code": "trouble_falling_or_staying_asleep" 
        },
        {
            #DPQ040: feeling_tired_or_having_little_energy 
            "question": "Over the last 2 weeks, how often have you been bothered by feeling tired or having little energy?",
            "options": ["Dummy","Not at all", "Several days", "More than half the days", "Nearly every day"],
            "var_code": "feeling_tired_or_having_little_energy" 
        },
        {
            #DPQ050: poor_appetitie_or_overeating 
            "question": "Over the last 2 weeks, how often have you been bothered by having poor appetite or overeating?",
            "options": ["Dummy","Not at all", "Several days", "More than half the days", "Nearly every day"],
            "var_code": "poor_appetitie_or_overeating" 
        },
        {
            #DPQ060: feeling_bad_about_yourself 
            "question": "Over the last 2 weeks, how often have you been bothered by feeling bad about yourself - or that you are a failure or have let yourself or your family down?",
            "options": ["Dummy","Not at all", "Several days", "More than half the days", "Nearly every day"],
            "var_code": "feeling_bad_about_yourself" 
        },
        {
            #DPQ070: trouble_concentrating 
            "question": "Over the last 2 weeks, how often have you been bothered by trouble concentrating on things, such as reading the newspaper or watching TV?",
            "options": ["Dummy","Not at all", "Several days", "More than half the days", "Nearly every day"],
            "var_code": "trouble_concentrating" 
        },
        {
            #DPQ080: moving_or_speaking_to_slowly_or_fast 
            "question": "Over the last 2 weeks, how often have you been bothered by moving or speaking so slowly that other people could have noticed? Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual?",
            "options": ["Dummy","Not at all", "Several days", "More than half the days", "Nearly every day"],
            "var_code": "moving_or_speaking_to_slowly_or_fast" 
        },
        {
            #DPQ090: thoughts_you_would_be_better_off_dead 
            "question": "Over the last 2 weeks, how often have you been bothered by thoughts that you would be better off dead or of hurting yourself in some way?",
            "options": ["Dummy","Not at all", "Several days", "More than half the days", "Nearly every day"],
            "var_code": "thoughts_you_would_be_better_off_dead" 
        },
        {
            #DPQ100: difficult_doing_daytoday_tasks 
            "question": "How difficult have these problems made it for you to do your work, take care of things at home, or get along with people?",
            "options": ["Dummy","Not at all difficult","Somewhat difficult","Very difficult","Extremely difficult"],
            "var_code": "difficult_doing_daytoday_tasks" 
        },
        {
            #ALQ290: times_with_12plus_alc 
            "question": "During the past 12 months, about how often did you have 12 or more drinks in a single day?",
            "options": ["Dummy","Never in the last year","Every day","Nearly every day","3-4 times a week","2 times a week","Once a week","2-3 times a month","Once a month", "7-11 times in the last year","3-6 times in the last year","1-2 times in the last year"],
            "var_code": "times_with_12plus_alc" 
        },
        {
            #BPQ080: high_cholesterol
            "question": "Have you ever been told by a health care professional that your blood cholesterol level was high?",
            "options": ["Dummy","Yes","No"],
            "var_code": "high_cholesterol" 
        },
        {
            #BPQ090D: cholesterol_prescription
            "question": "Have you ever been told by a health care professional to take prescribed medicine to lower blood cholesterol levels?",
            "options": ["Dummy","Yes","No"],
            "var_code": "cholesterol_prescription" 
        },
        {
            #BPQ020: high_bp
            "question": "Has a health professional ever told you that you have/had hypertension (high blood pressure)?",
            "options": ["Dummy","Yes","No"],
            "var_code": "high_bp" 
        },
        {
            #PAQ665: moderate_recreation
            "question": "In a typical week, do you do any moderate intensity recreational activities that cause a small increase in breathing/heart rate such as brisk walking, bicycling, or swimming for at least 10 minutes continuously?",
            "options": ["Dummy","Yes","No"],
            "var_code": "moderate_recreation" 
        },
        {
            #PAQ670: count_days_moderate_recreational_activity
            "question": "In a typical week, on how many days do you do moderate intensity recreational activities?",
            "options": ["Dummy","1","2","3","4","5","6","7"],
            "var_code": "count_days_moderate_recreational_activity" 
        },
        {
            #PAQ650: vigorous_recreation
            "question": "In a typical wekk, do you do any vigorous intensity recreational activities that cause large increases in breathing/heart rate like running or basketball for at least 10 minutes continuously?",
            "options": ["Dummy","Yes","No"],
            "var_code": "vigorous_recreation" 
        },
        {
            #MCQ160m: thyroid_issues
            "question": "Has a health professional ever told you that you had a thyroid problem?",
            "options": ["Dummy","Yes","No"],
            "var_code": "thyroid_issues" 
        },
        {
            #MCQ160a: arthritis
            "question": "Has a health professional ever told you that you have/had arthritis",
            "options": ["Dummy","Yes","No"],
            "var_code": "arthritis" 
        },
        {
            #MCQ160f: stroke
            "question": "Has a health professional ever told you that you had a stroke?",
            "options": ["Dummy","Yes","No"],
            "var_code": "stroke" 
        },
        {
            #MCQ010: asthma
            "question": "Has a health professional ever told you that you have asthma??",
            "options": ["Dummy","Yes","No"],
            "var_code": "asthma" 
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
            if question["question"] == "How old are you? (Please select the closest option)" or\
                question["question"] == "What is your height in inches? (Please select the closest option)" or \
                question["question"] == "What is your weight in pounds? (Please select the closest option)" or \
                question["question"] == "How many months ago did you have your baby? (Please select the closest option)":
                answer = st.selectbox(question["question"], question["options"])
                answers.append(answer)
            elif question["question"] == "About how long has it been since you last saw a health care professional about your health for any reason?" or\
                    question["question"] == "Over the last 2 weeks, how often have you been bothered by the following problems: little interest or pleasure in doing things?" or\
                    question["question"] == "Over the last 2 weeks, how often have you been bothered by feeling down, depressed, or hopeless?" or\
                    question["question"] == "Over the last 2 weeks, how often have you been bothered by trouble falling or staying asleep, or sleeping too much?" or\
                    question["question"] == "Over the last 2 weeks, how often have you been bothered by feeling tired or having little energy?" or\
                    question["question"] == "Over the last 2 weeks, how often have you been bothered by having poor appetite or overeating?" or\
                    question["question"] == "Over the last 2 weeks, how often have you been bothered by feeling bad about yourself - or that you are a failure or have let yourself or your family down?" or\
                    question["question"] == "Over the last 2 weeks, how often have you been bothered by trouble concentrating on things, such as reading the newspaper or watching TV?" or\
                    question["question"] == "Over the last 2 weeks, how often have you been bothered by moving or speaking so slowly that other people could have noticed? Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual?" or\
                    question["question"] == "Over the last 2 weeks, how often have you been bothered by thoughts that you would be better off dead or of hurting yourself in some way?" or\
                    question["question"] == "How difficult have these problems made it for you to do your work, take care of things at home, or get along with people?" or\
                    question["question"] == "During the past 12 months, about how often did you have 12 or more drinks in a single day?":
                answer = st.radio(question["question"], question["options"]) 
                answer_int = question["options"].index(answer)-1
                answers.append(answer_int)
            else:
                answer = st.radio(question["question"], question["options"])
                answer_int = question["options"].index(answer)
                answers.append(answer_int)

        # Generate predictions 
        feature_matrix = [answers]  # Convert answers to a feature matrix
        prediction = model.predict(feature_matrix)

        # Convert prediction label to description
        if prediction[0] == 1:
            prediction_text = "Predicted to have Postpartum Depression. We recommend reaching out to your healthcare provider for further evaluation."
        else:
            prediction_text = "Predicted not to have Postpartum Depression. Our model predicts a low likelihood of PPD but please reach out to your healthcare provider if you feel the need for further evaluation."

        submit = st.form_submit_button(label='Submit')

    # # Show the prediction after submission
    if submit:
        st.write("**Prediction:**", prediction_text)

    

if __name__ == "__main__":
    main()



