from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import os
from datetime import datetime
import uuid
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.layers import Layer
from src.preprocessing.palm_processor import PalmPreprocessor
import logging
from typing import Optional, Tuple, List
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Definisi custom layer
class AbsoluteDistance(Layer):
    def __init__(self, **kwargs):
        super(AbsoluteDistance, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.abs(inputs[0] - inputs[1])

    def compute_output_shape(self, input_shapes):
        return input_shapes[0]

    def get_config(self):
        config = super(AbsoluteDistance, self).get_config()
        return config

app = FastAPI(title="Palm Recognition System")

# Initialize preprocessor
preprocessor = PalmPreprocessor(target_size=(128, 128))

# Setup folders
base_dir = "data"
raw_dir = os.path.join(base_dir, "raw")
aug_dir = os.path.join(base_dir, "aug")
features_dir = os.path.join(base_dir, "features")

for directory in [raw_dir, aug_dir, features_dir]:
    os.makedirs(directory, exist_ok=True)

# Load Siamese Network model dengan custom objects
try:
    model = load_model('model/V2/siamese_model.h5', 
                      custom_objects={'AbsoluteDistance': AbsoluteDistance})
    logger.info("Siamese Network model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Siamese Network model: {str(e)}")
    raise RuntimeError("Could not initialize palm recognition model")

class PalmMatcher:
    def __init__(self, features_dir: str, distance_threshold: float = 0.7):
        self.features_dir = features_dir
        self.threshold = distance_threshold

    def extract_features(self, processed_image: np.ndarray) -> np.ndarray:
        """Extract feature vector from processed palm image using Siamese Network"""
        try:
            # Add batch dimension and get features from the model
            image_batch = np.expand_dims(processed_image, axis=0)
            features = model.predict(image_batch)
            return features[0]  # Return the feature vector (removing batch dimension)
        except Exception as e:
            logger.error(f"Feature extraction error: {str(e)}")
            raise

    def save_features(self, features: np.ndarray, user_id: str) -> None:
        """Save feature vector to .npy file"""
        feature_path = os.path.join(self.features_dir, f"{user_id}.npy")
        np.save(feature_path, features)

    def find_matching_palm(self, features: np.ndarray) -> Optional[str]:
        """Find matching palm by comparing features with stored feature vectors"""
        try:
            best_match = None
            min_distance = float('inf')

            # Compare with all stored features
            for feature_file in Path(self.features_dir).glob("*.npy"):
                stored_features = np.load(feature_file)
                distance = np.linalg.norm(features - stored_features)
                
                if distance < min_distance:
                    min_distance = distance
                    best_match = feature_file.stem  # user_id without .npy extension

            # Return match if distance is below threshold
            if min_distance < self.threshold:
                return best_match
            return None

        except Exception as e:
            logger.error(f"Palm matching error: {str(e)}")
            raise

# Initialize palm matcher
palm_matcher = PalmMatcher(features_dir)

async def process_palm_image(file: UploadFile) -> Tuple[np.ndarray, str]:
    """Process uploaded palm image and return processed image and user_id"""
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Generate user_id and save original image
        user_id = str(uuid.uuid4())
        original_path = os.path.join(raw_dir, f"{user_id}.jpg")
        cv2.imwrite(original_path, img)

        return img, user_id

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/register")
async def register(palm_image: UploadFile = File(...)):
    """
    Register new palm:
    1. Process and preprocess image
    2. Extract features using Siamese Network
    3. Check for existing matches
    4. Save features and create profile if no match found
    """
    try:
        # Process and preprocess image
        img, user_id = await process_palm_image(palm_image)
        processed_image = preprocessor.preprocess_image(img)

        if processed_image is None:
            raise HTTPException(status_code=400, detail="Failed to preprocess palm image")

        # Extract features
        features = palm_matcher.extract_features(processed_image)

        # Check if palm already exists
        existing_match = palm_matcher.find_matching_palm(features)
        if existing_match:
            raise HTTPException(
                status_code=400,
                detail="This palm is already registered in the system"
            )

        # Save features
        palm_matcher.save_features(features, user_id)

        # Generate and save augmentations
        augmented_images = preprocessor.generate_augmentations(processed_image)
        person_id = preprocessor.save_augmented_images(augmented_images, base_dir=base_dir)

        return JSONResponse(
            content={
                "status": "success",
                "message": "Registration successful",
                "user_id": user_id,
                "person_id": person_id,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/login")
async def login(palm_image: UploadFile = File(...)):
    """
    Login with palm:
    1. Process and preprocess image
    2. Extract features
    3. Find matching palm
    4. Return user profile if match found
    """
    try:
        # Process and preprocess image
        img, _ = await process_palm_image(palm_image)
        processed_image = preprocessor.preprocess_image(img)

        if processed_image is None:
            raise HTTPException(status_code=400, detail="Failed to process palm image")

        # Extract features and find match
        features = palm_matcher.extract_features(processed_image)
        matching_user_id = palm_matcher.find_matching_palm(features)

        if not matching_user_id:
            raise HTTPException(status_code=401, detail="No matching palm found")

        return JSONResponse(
            content={
                "status": "success",
                "message": "Login successful",
                "user_id": matching_user_id,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/show_profile")
async def show_profile(palm_image: UploadFile = File(...)):
    """
    Show profile using palm:
    1. Process and preprocess image
    2. Extract features
    3. Find matching palm
    4. Return associated profile
    """
    try:
        # Process and preprocess image
        img, _ = await process_palm_image(palm_image)
        processed_image = preprocessor.preprocess_image(img)

        if processed_image is None:
            raise HTTPException(status_code=400, detail="Failed to process palm image")

        # Extract features and find match
        features = palm_matcher.extract_features(processed_image)
        matching_user_id = palm_matcher.find_matching_palm(features)

        if not matching_user_id:
            raise HTTPException(status_code=404, detail="No matching profile found")

        # Here you would typically fetch the user's profile from your database
        # For now, we'll return a simple response
        return JSONResponse(
            content={
                "status": "success",
                "message": "Profile found",
                "user_id": matching_user_id,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Show profile error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)