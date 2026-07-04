import streamlit as st
import joblib

st.set_page_config(
    page_title="Emotion Detection",
    page_icon="😊",
    layout="centered"
)

with open("emotion_detection_pipline.pkl", "rb") as file:
    model = joblib.load(file)

st.title("😊 Emotion Detection from Text")
st.write("Enter any sentence and predict the emotion using NLP & Machine Learning.")


user_input = st.text_area(
    "Enter Text",
    height=180,
    placeholder="Example: I am very happy today!"
)

emotion_map = {
    1: "sadness",
    2: "anger",
    3: "love",
    4: "surprise",
    5: "fear",
    6: "joy"
}

if st.button("Predict Emotion"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:

        prediction = model.predict([user_input])[0]

        st.success(f"Predicted Emotion: **{emotion_map[prediction]}**")