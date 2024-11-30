from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import os
from datetime import datetime
import uuid
from src.preprocessing.palm_processor import PalmPreprocessor
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Testing Cuyy")

# Initialize preprocessor
preprocessor = PalmPreprocessor(target_size=(128, 128))

# Setup folders
base_dir = "data"
raw_dir = os.path.join(base_dir, "raw")
aug_dir = os.path.join(base_dir, "aug")

os.makedirs(raw_dir, exist_ok=True)
os.makedirs(aug_dir, exist_ok=True)


async def process_palm_image(file: UploadFile) -> tuple:
    """Process uploaded palm image"""
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Save original image
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
    Register new palm
    1. Process image
    2. Save original
    3. Generate augmentations
    4. Save all versions
    """
    try:
        # Process uploaded image
        img, user_id = await process_palm_image(palm_image)

        # Preprocess image
        processed_image = preprocessor.preprocess_image(img)

        if processed_image is None:
            raise HTTPException(
                status_code=400, detail="Failed to preprocess palm image"
            )

        # Generate augmentations
        augmented_images = preprocessor.generate_augmentations(processed_image)

        # Save augmented images
        person_id = preprocessor.save_augmented_images(
            augmented_images, base_dir=base_dir
        )

        return JSONResponse(
            content={
                "status": "success",
                "message": "Registration successful",
                "user_id": user_id,
                "person_id": person_id,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/login")
async def login(palm_image: UploadFile = File(...)):
    """
    Login with palm (currently only preprocessing)
    Note: Matching functionality will be added later
    """
    try:
        # Process and preprocess image
        img, _ = await process_palm_image(palm_image)
        processed_image = preprocessor.preprocess_image(img)

        if processed_image is None:
            raise HTTPException(status_code=400, detail="Failed to process palm image")

        return JSONResponse(
            content={
                "status": "success",
                "message": "Palm image processed successfully",
                "note": "Matching functionality will be added later",
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/show_profile")
async def show_profile(palm_image: UploadFile = File(...)):
    """
    Show profile using palm (currently only preprocessing)
    Note: Profile matching functionality will be added later
    """
    try:
        # Process and preprocess image
        img, _ = await process_palm_image(palm_image)
        processed_image = preprocessor.preprocess_image(img)

        if processed_image is None:
            raise HTTPException(status_code=400, detail="Failed to process palm image")

        return JSONResponse(
            content={
                "status": "success",
                "message": "Palm image processed successfully",
                "note": "Profile matching functionality will be added later",
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Show profile error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
