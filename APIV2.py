from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import os
from datetime import datetime
import uuid
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from src.preprocessing.palm_processor import PalmPreprocessor
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Buat folder penyimpanan
base_dir = "data"
raw_dir = os.path.join(base_dir, "raw")
aug_dir = os.path.join(base_dir, "aug")
features_dir = os.path.join(base_dir, "features")

for directory in [raw_dir, aug_dir, features_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

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

# Buat app FastAPI
app = FastAPI()

# Load preprocessor
preprocessor = PalmPreprocessor(target_size=(128, 128))

# Load model dengan custom objects
try:
    model = load_model('model/V2/siamese_model.h5', 
                      custom_objects={'AbsoluteDistance': AbsoluteDistance})
    print("Model berhasil dimuat")
except Exception as e:
    print(f"Error memuat model: {str(e)}")
    raise RuntimeError("Gagal memuat model")

def extract_features(image):
    """Fungsi untuk ekstrak fitur dari gambar"""
    try:
        # Tambah dimensi batch dan channel (RGB)
        if len(image.shape) == 2:  # jika grayscale
            image = np.expand_dims(image, axis=-1)
            image = np.repeat(image, 3, axis=-1)
        
        image_batch = np.expand_dims(image, axis=0)
        
        # Siamese network butuh 2 input yang sama untuk ekstrak fitur
        # Kita gunakan gambar yang sama sebagai anchor dan target
        features = model.predict([image_batch, image_batch])
        return features[0]
        
    except Exception as e:
        print(f"Error ekstrak fitur: {str(e)}")
        raise

def find_matching_palm(features, threshold=0.7):
    """Fungsi untuk mencari palm yang cocok"""
    try:
        best_match = None
        min_distance = float('inf')
        
        # Loop semua file fitur yang ada
        for filename in os.listdir(features_dir):
            if filename.endswith('.npy'):
                # Load fitur yang tersimpan
                stored_features = np.load(os.path.join(features_dir, filename))
                # Hitung jarak
                distance = np.linalg.norm(features - stored_features)
                
                # Update best match jika jarak lebih kecil
                if distance < min_distance:
                    min_distance = distance
                    best_match = filename.replace('.npy', '')
        
        # Return hasil jika di bawah threshold
        if min_distance < threshold:
            return best_match
        return None
        
    except Exception as e:
        print(f"Error matching palm: {str(e)}")
        return None

@app.post("/register")
async def register(palm_image: UploadFile = File(...)):
    try:
        # Baca gambar yang diupload
        contents = await palm_image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"error": "Gambar tidak valid"}
        
        # Generate ID dan simpan gambar asli
        user_id = str(uuid.uuid4())
        cv2.imwrite(os.path.join(raw_dir, f"{user_id}.jpg"), img)
        
        # Preprocessing gambar
        processed_image = preprocessor.preprocess_image(img)
        if processed_image is None:
            return {"error": "Gagal preprocessing gambar"}
        
        # Ekstrak fitur
        features = extract_features(processed_image)
        if features is None:
            return {"error": "Gagal ekstrak fitur"}
        
        # Cek apakah palm sudah terdaftar
        existing_match = find_matching_palm(features)
        if existing_match:
            return {"error": "Palm sudah terdaftar"}
        
        # Simpan fitur
        np.save(os.path.join(features_dir, f"{user_id}.npy"), features)
        
        # Buat dan simpan augmentasi
        augmented_images = preprocessor.generate_augmentations(processed_image)
        person_id = preprocessor.save_augmented_images(augmented_images, base_dir=base_dir)
        
        return {
            "status": "success",
            "message": "Registrasi berhasil",
            "user_id": user_id,
            "person_id": person_id
        }
        
    except Exception as e:
        print(f"Error registrasi: {str(e)}")
        return {"error": str(e)}

@app.post("/login")
async def login(palm_image: UploadFile = File(...)):
    try:
        # Baca gambar
        contents = await palm_image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"error": "Gambar tidak valid"}
        
        # Preprocessing gambar
        processed_image = preprocessor.preprocess_image(img)
        if processed_image is None:
            return {"error": "Gagal preprocessing gambar"}
        
        # Ekstrak fitur dan cari yang cocok
        features = extract_features(processed_image)
        matching_user_id = find_matching_palm(features)
        
        if not matching_user_id:
            return {"error": "Palm tidak dikenali"}
        
        return {
            "status": "success",
            "message": "Login berhasil",
            "user_id": matching_user_id
        }
        
    except Exception as e:
        print(f"Error login: {str(e)}")
        return {"error": str(e)}

@app.post("/show_profile")
async def show_profile(palm_image: UploadFile = File(...)):
    try:
        # Baca gambar
        contents = await palm_image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"error": "Gambar tidak valid"}
        
        # Preprocessing gambar
        processed_image = preprocessor.preprocess_image(img)
        if processed_image is None:
            return {"error": "Gagal preprocessing gambar"}
        
        # Ekstrak fitur dan cari yang cocok
        features = extract_features(processed_image)
        matching_user_id = find_matching_palm(features)
        
        if not matching_user_id:
            return {"error": "Profil tidak ditemukan"}
            
        # Buat data profil sederhana
        profile = {
            "user_id": matching_user_id,
            "raw_image": f"data/raw/{matching_user_id}.jpg",
            "aug_folder": f"data/aug/{matching_user_id}/",
            "registration_date": datetime.fromtimestamp(
                os.path.getctime(
                    os.path.join(features_dir, f"{matching_user_id}.npy")
                )
            ).strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return {
            "status": "success",
            "message": "Profil ditemukan",
            "profile": profile
        }
        
    except Exception as e:
        print(f"Error show profile: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)