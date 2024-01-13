import io
import os
import string

from keras.models import load_model
import PIL
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, HTTPException
from starlette.responses import RedirectResponse
from pydantic import BaseModel
from PIL import Image


from config.settings import models_dir


class Prediction(BaseModel):
    prediction: str


#MODEL_PATH = os.path.join(models_dir, 'cnn')
MODEL_PATH = '/Users/teodorastereciu/PycharmProjects/mlp/models/final_model.h5'

#model = tf.saved_model.load(MODEL_PATH)
model = load_model(MODEL_PATH)

app = FastAPI(
    title="ASL sign recognition",
    summary="An API endpoint to classify ASL signs.",
    description="",
    version="alpha")

categories = {i: letter for i, letter in enumerate(char for char in string.ascii_uppercase if char not in {'J', 'Z'})}


@app.get("/", description="API endpoint that redirects to documentation")
async def root():
    return RedirectResponse(url='/docs')


def process_image(image):
    image_data = image.file.read()
    image = Image.open(io.BytesIO(image_data))
    gray = image.convert('L')
    resized = gray.resize((200, 200))
    image_arr = np.array(resized)
    normalized = image_arr / 255.0
    ready = normalized.reshape((1, *normalized.shape))
    print("Ready: ", ready)
    return ready


@app.post("/predict", description="", response_model=Prediction, response_description="")
async def predict(image: UploadFile):
    try:
        processed_image = process_image(image)
    except PIL.UnidentifiedImageError:
        raise HTTPException(status_code=415, detail="Invalid image")
    prediction = model.predict(processed_image)
    print("Prediction: ", prediction)
    predicted_class_index = np.argmax(prediction, axis=1)
    print("Predict class idx: ", predicted_class_index)
    predicted_label = categories[predicted_class_index.item()]
    print("Predict label: ", predicted_label)
    return Prediction(prediction=predicted_label)
