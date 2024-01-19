import io
import os
import string
from pathlib import Path

from keras.models import load_model
import PIL
import numpy as np
from fastapi import FastAPI, UploadFile, HTTPException
from starlette.responses import RedirectResponse
from pydantic import BaseModel
from PIL import Image


class Prediction(BaseModel):
    """
    Predictions are strings.
    """
    prediction: str


BASE_DIR = Path(__file__).resolve(strict=True).parent
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'final_model.h5')

model = load_model(MODEL_PATH)

app = FastAPI(
    title="ASL Alphabet Sign Recognition",
    description="An API endpoint to classify ASL alphabet signs.",
    version="1.0")

categories = {i: letter for i, letter in enumerate(char for char in string.ascii_uppercase if char not in {'J', 'Z'})}


@app.get("/", description="API endpoint that redirects to documentation")
async def root():
    return RedirectResponse(url='/docs')


def process_image(image: UploadFile):
    """
    Process the uploaded image for ASL sign recognition.

    Parameters:
    - `image`: The uploaded image file.

    Returns:
    - `np.ndarray`: The processed image array.

    Raises:
    - `HTTPException 415`: If the image cannot be identified, loaded, or has insufficient dimensions.
    """
    try:
        image_data = image.file.read()
        image = Image.open(io.BytesIO(image_data))
        if image.width < 200 or image.height < 200:
            raise HTTPException(status_code=415, detail="Image dimensions should be at least (200, 200)")
        gray = image.convert('L')
        resized = gray.resize((200, 200))
        image_arr = np.array(resized)
        normalized = image_arr / 255.0
        ready = normalized.reshape((1, *normalized.shape))
        return ready
    except PIL.UnidentifiedImageError:
        raise HTTPException(status_code=415, detail="Invalid image")


@app.post("/predict", description="Get predictions for ASL alphabet signs", response_model=Prediction, response_description="Predicted letter.")
async def predict(image: UploadFile):
    """
    Predict the ASL alphabet sign from a JPG image.

    Parameters:
    - `image`: The absolute path to the image file to be processed.

    Returns:
    - `Prediction`: The predicted letter in a JSON string.

    Raises:
    - `HTTPException 415`: If the uploaded file is not a valid image.

    Example usage:
    ```bash
    curl -X POST "http://127.0.0.1:8000/predict" \
    -H "accept: application/json" \
    -H "Content-Type: multipart/form-data" \
    -F "image=@/path/to/your/image.jpg"
    ```
    """
    try:
        processed_image = process_image(image)
    except PIL.UnidentifiedImageError:
        raise HTTPException(status_code=415, detail="Invalid image")
    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction, axis=1)
    predicted_label = categories[predicted_class_index.item()]
    return Prediction(prediction=predicted_label)
