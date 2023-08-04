import streamlit as st
from operator import mod
from os import getcwd
from os.path import exists, join

import joblib
import pandas as pd
import numpy as np

import json
import pickle
from collections import Counter
import pydeck as pdk
import smtplib
import requests


hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """

st.set_page_config(page_title="Survey Model", page_icon="./page_logo.png")

st.markdown(hide_default_format, unsafe_allow_html=True)

st.image("./header_survey.png")


st.sidebar.header("Survey Model")
st.write(
    """This generator asks questions predetermined to be relevant for predicting depression and uses responses in our trained model to predict depression."""
)

st.write(
    "Please view our privacy policy for details regarding data prior to using the models."
)


def PostRequestSurveyAPI(answer_dict):
    url = "http://network-load-balancer-3ec3c60f32bd38c8.elb.us-west-1.amazonaws.com/predict"
    myobj = {"surveys": [answer_dict]}
    print(myobj)
    try:
        x = requests.post(url, json=myobj)
        print("request successful")
        print(x.json())

        prediction = x.json()["predictions"][0]["prediction"]
        print(f"prediction is {prediction}")
    except Exception as e:
        print(e)
        prediction = 0
    return prediction


# Main function
def main():
    # Provide the questions and answer options
    questions = [
        {
            # RIDAGEYR: age_in_years
            "question": "How old are you? (Please select the closest option)",
            "options": list(range(0, 86)),
            "option_is_continuous": True,
            "var_code": "age_in_years",
        },
        {
            # height: height_in for weight_lbs_over_height_in_ratio
            "question": "What is your height in inches? (Please select the closest option)",
            "options": list(range(48, 82)),
            "option_is_continuous": True,
            "var_code": "height_in",
        },
        {
            # weight: weight_lbs for weight_lbs_over_height_in_ratio
            "question": "What is your weight in pounds? (Please select the closest option)",
            "options": list(range(75, 494)),
            "option_is_continuous": True,
            "var_code": "weight_lbs",
        },
        {
            # DMDBORN4: is_usa_born
            "question": "Were you born in the United States?",
            "options": ["Dummy", "Yes", "No"],
            "option_codes": [None, 1, 2],
            "var_code": "is_usa_born",
        },
        {
            # HIQ011: have_health_insurance
            "question": "Are you covered by health insurance or some kind of health care plan?",
            "options": ["Dummy", "Yes", "No"],
            "option_codes": [None, 1, 2],
            "var_code": "have_health_insurance",
        },
        {
            # RHQ197: months_since_birth
            "question": "How many months ago did you have your baby? (Please select the closest option)",
            "options": list(range(1, 28)),
            "option_is_continuous": True,
            "var_code": "months_since_birth",
        },
        {
            # RHQ031: regular_periods
            "question": "Have you had at least one menstrual period in the past 12 months?",
            "options": ["Dummy", "Yes", "No"],
            "option_codes": [None, 1, 2],
            "var_code": "regular_periods",
        },
        {
            # RHQ540: horomones_not_bc
            "question": "Have you ever used female hormones such as estrogen and progesterone? Please include any forms such as pills, cream, patch, and inejctables, but do not include birth controls methods or use for infertility.",
            "options": ["Dummy", "Yes", "No"],
            "option_codes": [None, 1, 2],
            "var_code": "horomones_not_bc",
        },
        {
            # HUD062: time_since_last_healthcare
            "question": "About how long has it been since you last saw a health care professional about your health for any reason?",
            "options": [
                "Dummy",
                "Never",
                "Within the past year",
                "Within the last 2 years",
                "Within the last 5 years",
                "5 years ago or more",
            ],
            "option_codes": [None, 0, 1, 2, 3, 4],
            "var_code": "time_since_last_healthcare",
        },
        {
            # HUQ090: seen_mental_health_professional
            "question": "During the past 12 months, have you seen or talked to a mental health professional about your health?",
            "options": ["Dummy", "Yes", "No"],
            "option_codes": [None, 1, 2],
            "var_code": "seen_mental_health_professional",
        },
        {
            # DPQ010: little_interest_in_doing_things
            "question": "Over the last 2 weeks, how often have you been bothered by the following problems: little interest or pleasure in doing things?",
            "options": [
                "Dummy",
                "Not at all",
                "Several days",
                "More than half the days",
                "Nearly every day",
            ],
            "option_codes": [None, 0, 1, 2, 3],
            "var_code": "little_interest_in_doing_things",
        },
        {
            # DPQ020: feeling_down_depressed_hopeless
            "question": "Over the last 2 weeks, how often have you been bothered by feeling down, depressed, or hopeless?",
            "options": [
                "Dummy",
                "Not at all",
                "Several days",
                "More than half the days",
                "Nearly every day",
            ],
            "option_codes": [None, 0, 1, 2, 3],
            "var_code": "feeling_down_depressed_hopeless",
        },
        {
            # DPQ030: trouble_falling_or_staying_asleep
            "question": "Over the last 2 weeks, how often have you been bothered by trouble falling or staying asleep, or sleeping too much?",
            "options": [
                "Dummy",
                "Not at all",
                "Several days",
                "More than half the days",
                "Nearly every day",
            ],
            "option_codes": [None, 0, 1, 2, 3],
            "var_code": "trouble_falling_or_staying_asleep",
        },
        {
            # DPQ040: feeling_tired_or_having_little_energy
            "question": "Over the last 2 weeks, how often have you been bothered by feeling tired or having little energy?",
            "options": [
                "Dummy",
                "Not at all",
                "Several days",
                "More than half the days",
                "Nearly every day",
            ],
            "option_codes": [None, 0, 1, 2, 3],
            "var_code": "feeling_tired_or_having_little_energy",
        },
        {
            # DPQ050: poor_appetitie_or_overeating
            "question": "Over the last 2 weeks, how often have you been bothered by having poor appetite or overeating?",
            "options": [
                "Dummy",
                "Not at all",
                "Several days",
                "More than half the days",
                "Nearly every day",
            ],
            "option_codes": [None, 0, 1, 2, 3],
            "var_code": "poor_appetitie_or_overeating",
        },
        {
            # DPQ060: feeling_bad_about_yourself
            "question": "Over the last 2 weeks, how often have you been bothered by feeling bad about yourself - or that you are a failure or have let yourself or your family down?",
            "options": [
                "Dummy",
                "Not at all",
                "Several days",
                "More than half the days",
                "Nearly every day",
            ],
            "option_codes": [None, 0, 1, 2, 3],
            "var_code": "feeling_bad_about_yourself",
        },
        {
            # DPQ070: trouble_concentrating
            "question": "Over the last 2 weeks, how often have you been bothered by trouble concentrating on things, such as reading the newspaper or watching TV?",
            "options": [
                "Dummy",
                "Not at all",
                "Several days",
                "More than half the days",
                "Nearly every day",
            ],
            "option_codes": [None, 0, 1, 2, 3],
            "var_code": "trouble_concentrating",
        },
        {
            # DPQ080: moving_or_speaking_to_slowly_or_fast
            "question": "Over the last 2 weeks, how often have you been bothered by moving or speaking so slowly that other people could have noticed? Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual?",
            "options": [
                "Dummy",
                "Not at all",
                "Several days",
                "More than half the days",
                "Nearly every day",
            ],
            "option_codes": [None, 0, 1, 2, 3],
            "var_code": "moving_or_speaking_to_slowly_or_fast",
        },
        {
            # DPQ090: thoughts_you_would_be_better_off_dead
            "question": "Over the last 2 weeks, how often have you been bothered by thoughts that you would be better off dead or of hurting yourself in some way?",
            "options": [
                "Dummy",
                "Not at all",
                "Several days",
                "More than half the days",
                "Nearly every day",
            ],
            "option_codes": [None, 0, 1, 2, 3],
            "var_code": "thoughts_you_would_be_better_off_dead",
        },
        {
            # DPQ100: difficult_doing_daytoday_tasks
            "question": "How difficult have these problems made it for you to do your work, take care of things at home, or get along with people?",
            "options": [
                "Dummy",
                "Not at all difficult",
                "Somewhat difficult",
                "Very difficult",
                "Extremely difficult",
            ],
            "option_codes": [None, 0, 1, 2, 3],
            "var_code": "difficult_doing_daytoday_tasks",
        },
        {
            # ALQ290: times_with_12plus_alc
            "question": "During the past 12 months, about how often did you have 12 or more drinks in a single day?",
            "options": [
                "Dummy",
                "Never in the last year",
                "Every day",
                "Nearly every day",
                "3-4 times a week",
                "2 times a week",
                "Once a week",
                "2-3 times a month",
                "Once a month",
                "7-11 times in the last year",
                "3-6 times in the last year",
                "1-2 times in the last year",
            ],
            "option_codes": [None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "var_code": "times_with_12plus_alc",
        },
        {
            # BPQ080: high_cholesterol
            "question": "Have you ever been told by a health care professional that your blood cholesterol level was high?",
            "options": ["Dummy", "Yes", "No"],
            "option_codes": [None, 1, 2],
            "var_code": "high_cholesterol",
        },
        {
            # BPQ090D: cholesterol_prescription
            "question": "Have you ever been told by a health care professional to take prescribed medicine to lower blood cholesterol levels?",
            "options": ["Dummy", "Yes", "No"],
            "option_codes": [None, 1, 2],
            "var_code": "cholesterol_prescription",
        },
        {
            # BPQ020: high_bp
            "question": "Has a health professional ever told you that you have/had hypertension (high blood pressure)?",
            "options": ["Dummy", "Yes", "No"],
            "option_codes": [None, 1, 2],
            "var_code": "high_bp",
        },
        {
            # PAQ665: moderate_recreation
            "question": "In a typical week, do you do any moderate intensity recreational activities that cause a small increase in breathing/heart rate such as brisk walking, bicycling, or swimming for at least 10 minutes continuously?",
            "options": ["Dummy", "Yes", "No"],
            "option_codes": [None, 1, 2],
            "var_code": "moderate_recreation",
        },
        {
            # PAQ670: count_days_moderate_recreational_activity
            "question": "In a typical week, on how many days do you do moderate intensity recreational activities?",
            "options": ["Dummy", "1", "2", "3", "4", "5", "6", "7"],
            "option_codes": [None, 1, 2, 3, 4, 5, 6, 7],
            "var_code": "count_days_moderate_recreational_activity",
        },
        {
            # PAQ650: vigorous_recreation
            "question": "In a typical week, do you do any vigorous intensity recreational activities that cause large increases in breathing/heart rate like running or basketball for at least 10 minutes continuously?",
            "options": ["Dummy", "Yes", "No"],
            "option_codes": [None, 1, 2],
            "var_code": "vigorous_recreation",
        },
        {
            # MCQ160m: thyroid_issues
            "question": "Has a health professional ever told you that you had a thyroid problem?",
            "options": ["Dummy", "Yes", "No"],
            "option_codes": [None, 1, 2],
            "var_code": "thyroid_issues",
        },
        {
            # MCQ160a: arthritis
            "question": "Has a health professional ever told you that you have/had arthritis",
            "options": ["Dummy", "Yes", "No"],
            "option_codes": [None, 1, 2],
            "var_code": "arthritis",
        },
        {
            # MCQ160f: stroke
            "question": "Has a health professional ever told you that you had a stroke?",
            "options": ["Dummy", "Yes", "No"],
            "option_codes": [None, 1, 2],
            "var_code": "stroke",
        },
        {
            # MCQ010: asthma
            "question": "Has a health professional ever told you that you have asthma??",
            "options": ["Dummy", "Yes", "No"],
            "option_codes": [None, 1, 2],
            "var_code": "asthma",
        },
        # TODO: need to add actual questions for below
        {
            # : count_lost_10plus_pounds
            "question": "fill question4",
            "options": ["Dummy", "Yes", "No"],
            "option_codes": [None, 1, 2],
            "var_code": "count_lost_10plus_pounds",
        },
        {
            # : times_with_8plus_alc
            "question": "During the past 12 months, about how often did you have 8 or more drinks in a single day?",
            "options": [
                "Dummy",
                "Never in the last year",
                "Every day",
                "Nearly every day",
                "3-4 times a week",
                "2 times a week",
                "Once a week",
                "2-3 times a month",
                "Once a month",
                "7-11 times in the last year",
                "3-6 times in the last year",
                "1-2 times in the last year",
            ],
            "option_codes": [None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "var_code": "times_with_8plus_alc",
        },
        {
            # : duration_last_healthcare_visit
            "question": "fill question6",
            "options": ["Dummy", "Yes", "No"],
            "option_codes": [None, 1, 2],
            "var_code": "duration_last_healthcare_visit",
        },
        {
            # : work_schedule
            "question": "fill question7",
            "options": ["Dummy", "Yes", "No"],
            "option_codes": [None, 1, 2],
            "var_code": "work_schedule",
        },
    ]

    # Display the questions and collect answers in a form
    with st.form(key="survey_form"):
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
        # dictionary of question var_code:option_codes
        # we pass this dictionary to the predict api via post request
        survey_answers_dict = {}

        # Present questions with answer options
        for question in questions:
            if (
                question["question"]
                == "How old are you? (Please select the closest option)"
                or question["question"]
                == "What is your height in inches? (Please select the closest option)"
                or question["question"]
                == "What is your weight in pounds? (Please select the closest option)"
                or question["question"]
                == "How many months ago did you have your baby? (Please select the closest option)"
            ):
                answer = st.selectbox(question["question"], question["options"])
                # answers.append(answer)
            elif (
                question["question"]
                == "About how long has it been since you last saw a health care professional about your health for any reason?"
                or question["question"]
                == "Over the last 2 weeks, how often have you been bothered by the following problems: little interest or pleasure in doing things?"
                or question["question"]
                == "Over the last 2 weeks, how often have you been bothered by feeling down, depressed, or hopeless?"
                or question["question"]
                == "Over the last 2 weeks, how often have you been bothered by trouble falling or staying asleep, or sleeping too much?"
                or question["question"]
                == "Over the last 2 weeks, how often have you been bothered by feeling tired or having little energy?"
                or question["question"]
                == "Over the last 2 weeks, how often have you been bothered by having poor appetite or overeating?"
                or question["question"]
                == "Over the last 2 weeks, how often have you been bothered by feeling bad about yourself - or that you are a failure or have let yourself or your family down?"
                or question["question"]
                == "Over the last 2 weeks, how often have you been bothered by trouble concentrating on things, such as reading the newspaper or watching TV?"
                or question["question"]
                == "Over the last 2 weeks, how often have you been bothered by moving or speaking so slowly that other people could have noticed? Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual?"
                or question["question"]
                == "Over the last 2 weeks, how often have you been bothered by thoughts that you would be better off dead or of hurting yourself in some way?"
                or question["question"]
                == "How difficult have these problems made it for you to do your work, take care of things at home, or get along with people?"
                or question["question"]
                == "During the past 12 months, about how often did you have 12 or more drinks in a single day?"
            ):
                answer = st.radio(question["question"], question["options"])
            else:
                answer = st.radio(question["question"], question["options"])
            if question.get("option_is_continuous") == True:
                survey_answers_dict[question.get("var_code")] = answer
            else:
                # use the index pf the answer to get corresponding code
                if question.get("option_codes"):
                    survey_answers_dict[question.get("var_code")] = question.get(
                        "option_codes"
                    )[question["options"].index(answer)]

        # remove nulls convert to float
        survey_answers_dict = {
            k: (float(v) if v else 0.0) for k, v in survey_answers_dict.items()
        }
        # survey_answers_dict = {
        #     k: (float(v) if v else None) for k, v in survey_answers_dict.items()
        # }
        # Make prediction using the model using API
        prediction = st.form_submit_button(
            label="Submit", on_click=PostRequestSurveyAPI, args=(survey_answers_dict,)
        )

        # Convert prediction label to description
        if prediction == 1:
            prediction_text = "Predicted to have Postpartum Depression. We recommend reaching out to your healthcare provider for further evaluation."
        else:
            prediction_text = "Predicted not to have Postpartum Depression. Our model predicts a low likelihood of PPD but please reach out to your healthcare provider if you feel the need for further evaluation."

        # # Show the prediction after submission
        if prediction:
            st.write("**Prediction:**", prediction_text)


if __name__ == "__main__":
    main()
