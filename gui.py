import os
from pathlib import Path

from keras.models import load_model
import streamlit as st
import string
from PIL import Image
import numpy as np

import warnings

warnings.filterwarnings("ignore")

# set some pre-defined configurations for the page,
# such as the page title, logo-icon, page loading state
# (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    #page_title="Mango Leaf Disease Detection",
    #page_icon=":mango:",
    initial_sidebar_state='auto'
)

# hide the part of the code, as this is just for adding some custom CSS styling but not a part of the main idea
hide_streamlit_style = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style,
            unsafe_allow_html=True)

categories = {i: letter for i, letter in enumerate(char for char in string.ascii_uppercase if char not in {'J', 'Z'})}


def prediction_label(prediction):
    predicted_class_index = np.argmax(prediction, axis=1)
    return categories[predicted_class_index.item()]


with st.sidebar:
    st.image('bg.jpeg')
    st.title("Try it out!")
    st.subheader(
        "Automatic recognition of ASL alphabet signs\n"
        " Upload a picture and get a prediction!")

st.write("""
         # ASL Alphabet Recognition
         """
         )


@st.cache_data
def keras_predict(input, _model):
    return model.predict(input)


def import_and_predict(image, model):
    gray = image.convert('L')
    resized = gray.resize((200, 200))
    image_arr = np.array(resized)
    normalized = image_arr / 255.0
    ready = normalized.reshape((1, *normalized.shape))
    return keras_predict(ready, model)


BASE_DIR = Path(__file__).resolve(strict=True).parent
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'final_model.h5')


@st.cache_resource
def load_keras_model():
    return load_model(MODEL_PATH)


with st.spinner('Model is being loaded..'):
    model = load_keras_model()

if 'uploader_disabled' not in st.session_state:
    st.session_state.uploader_disabled = False
if 'camera_disabled' not in st.session_state:
    st.session_state.camera_disabled = False


def update_uploader():
    st.session_state.uploader_disabled = not st.session_state.uploader_disabled


def update_camera():
    st.session_state.camera_disabled = not st.session_state.camera_disabled


file = st.file_uploader("", type=["jpg", "png"],
                        on_change=update_camera,
                        disabled=st.session_state.uploader_disabled)

if file is None:
    st.text("Please upload an image file")
else:
    st.session_state.camera_disabled = True
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    if prediction is not None:
        st.balloons()
        st.sidebar.write(f"## {prediction_label(prediction)}")

picture = st.camera_input("Or.. take a picture!",
                          on_change=update_uploader,
                          disabled=st.session_state.camera_disabled)

if picture is not None:
    # st.session_state.uploader_disabled = True
    st.image(picture, use_column_width=True)
    image = Image.open(picture)
    prediction = import_and_predict(image, model)
    if prediction is not None:
        st.balloons()
        st.sidebar.write(f"## {prediction_label(prediction)}")
