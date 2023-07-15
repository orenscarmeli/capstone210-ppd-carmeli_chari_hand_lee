import streamlit as st
import json
import bcrypt
import re

def main():
    st.title("Sign in or Register")

    registered = st.checkbox("Already registered? Sign in")
    if registered: #display log in form if already registered
        login()
    else:          #display registration form if user isn't registered
        register()

def register():
    st.write("Please create your account.")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Register"):
        if password == confirm_password:
            # Check if the username already exists
            if username_exists(username):
                st.error("Username already exists. Please choose a different username.")
            else:
                # Check password complexity requirements
                if not is_password_complex(password):
                    st.error("Password must be at least 8 characters long and contain a number and special character.")
                else:
                    # Hash the password using bcrypt
                    hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

                    # Store the username and hashed password in a JSON file
                    user_data = {
                        "username": username,
                        "hashed_password": hashed_password.decode("utf-8")
                    }
                    with open("user_credentials.json", "w") as f:
                        json.dump(user_data, f)
                    st.write("Registration successful. Please sign in.")
        else:
            st.error("Passwords do not match. Please try again.")

def login():
    st.write("Please sign in to your account.")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        # Read credentials from JSON file
        with open("user_credentials.json", "r") as f:
            user_data = json.load(f)
        
        # Confirm login credentials against the stored values
        hashed_password = user_data.get("hashed_password", "")

        if bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8")):
            st.write("Login successful.")
            redirect_page()
        else:
            st.error("Invalid username or password.")

def username_exists(username):
    # Read credentials from JSON file
    try:
        with open("user_credentials.json", "r") as f:
            user_data = json.load(f)
            return username == user_data.get("username", "")
    except FileNotFoundError:
        return False

def is_password_complex(password):
    # Check password complexity requirements
    if len(password) < 8:
        return False
    if not re.search(r"\d", password):
        return False
    if not re.search(r"[ !#$%&'()*+,-./[\\\]^_`{|}~" + r'"]', password):
        return False
    return True

def redirect_page():
    redirect_url = "http://localhost:8502/Survey_Model"  # Modify the URL path according to your app's routes
    st.markdown(f"Redirect to the survey app page by clicking [here]({redirect_url}).")


if __name__ == "__main__":
    main()