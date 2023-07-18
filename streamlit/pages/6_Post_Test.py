import streamlit as st
import requests

st.set_page_config(page_title="Prediction generator", page_icon="")

st.markdown("# Prediction generator")
st.sidebar.header("Prediction generator")
st.write(
    """This generator asks questions predetermined to be relevant for predicting depression and uses responses in our trained model to predict depression"""
)

def PostRequestSurveyAPI(answer_dict):

    url = 'http://network-load-balancer-3ec3c60f32bd38c8.elb.us-west-1.amazonaws.com/predict'
    myobj = {
        "surveys" : [ 
           answer_dict
        ]
    }
    x = requests.post(url, json=myobj)
    print('request successful')
    print(x.json())
    # get the response
    # prediction_response = x.json()['predictions'][0]['prediction']
    # get into json response to get 0 or 1 prediction
    prediction = x.json()['predictions'][0]['prediction']
    print(f'prediction is {prediction}')
    return prediction

# Main function
def main():

    # st.title("Postpartum Depression Screening")
    # st.write("Answer the following questions to predict postpartum depression")

    # Define the questions and answer options

    questions = [
        {
            "question": "Over the last 2 weeks, how often have you been bothered by the following problems: little interest or pleasure in doing things?",
            "var_code": "DPQ010",
            "var_name": "little_interest_in_doing_things",
            "options": ["Not at all", "Several days", "More than half the days", "Nearly every day"]
        },
        {
            "question": "Over the last 2 weeks, how often have you been bothered by feeling down, depressed, or hopeless?",
            "var_code": "DPQ020",
            "var_name": "feeling_down_depressed_hopeless",
            "options": ["Not at all", "Several days", "More than half the days", "Nearly every day"]
        },
        {
            "question": "Over the last 2 weeks, how often have you been bothered by trouble falling or staying asleep, or sleeping too much?",
            "var_code": "DPQ030",
            # TODO: need to change API to remove extra forbid for dev purposes
            # this variable was not in the model i picked for the API dev so will throw an error
            # "var_name": "trouble_falling_or_staying_asleep",
            # can just name to a column we know we have for testing
            "var_name": "feeling_down_depressed_hopeless",
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
        answer_dict = {}

        for question in questions:
            answer = st.radio(question["question"], question["options"])
            answer_int = question["options"].index(answer)
            answer_dict[question['var_name']] = answer_int
            answers.append(answer_int)

        # Make prediction using the model using API
        prediction = st.form_submit_button(label='Submit', on_click=PostRequestSurveyAPI, args=(answer_dict, ))
        

    # Show the prediction after submission
    if prediction is not None:
        # Convert prediction label to description
        if prediction == 1:
            prediction_text = "Likely to have Postpartum Depression"
        else:
            prediction_text = "Not likely to have Postpartum Depression"
        st.write("**Prediction:**", prediction_text)




# Run the app
if __name__ == "__main__":
    main()