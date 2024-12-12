"""
Model Palm print recognizer menggunakan arsitektur Siamese neural network
==========================================================

File ini berisi implementasi model pengenalan telapak tangan menggunakan arsitektur jaringan Siamese.
Jaringan Siamese adalah arsitektur neural network yang terdiri dari dua sub-jaringan identik yang
berbagi bobot yang sama. Model ini digunakan untuk membandingkan kemiripan antara dua gambar
telapak tangan.

Fitur Utama:
------------
1. Preprocessing gambar menggunakan PalmPreprocessor
2. Ekstraksi fitur menggunakan Convolutional Neural Network (CNN)
3. Normalisasi L2 pada embedding layer
4. Perhitungan similarity menggunakan jarak Euclidean
5. Penyimpanan dan pemulihan database embedding

Arsitektur Model:
----------------
- Input: Gambar grayscale 128x128 piksel
- Convolutional layers:
  * Layer 1: 64 filter, kernel 10x10
  * Layer 2: 128 filter, kernel 7x7
  * Layer 3: 128 filter, kernel 4x4
  * Layer 4: 256 filter, kernel 4x4
- MaxPooling setelah setiap conv layer
- Dense layer: 4096 neuron dengan aktivasi sigmoid
- L2 normalization pada output

Dependencies:
------------
- tensorflow: Framework deep learning
- numpy: Operasi array dan matriks
- opencv-python: Pengolahan gambar
- json: Penyimpanan database
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Lambda
import cv2
import os
import json
from typing import Dict, List, Tuple
import logging
from preprocessing.palm_processorV5 import PalmPreprocessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def euclidean_distance(vects):
    """
    Menghitung jarak Euclidean antara dua vektor.
    
    Args:
        vects: Tuple dari dua vektor yang akan dibandingkan
        
    Returns:
        Jarak Euclidean antara kedua vektor
    """
    x, y = vects
    sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_square, tf.keras.backend.epsilon()))

def contrastive_loss(y_true, y_pred, margin=1.0):
    """
    Contrastive loss function.
    
    Args:
        y_true: Label (1 for same pairs, 0 for different pairs)
        y_pred: Predicted distance
        margin: Margin for negative pairs
    """
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

def build_feature_extractor(input_shape=(128, 128, 1)):
    """
    Membangun CNN untuk ekstraksi fitur dari gambar telapak tangan.
    
    Args:
        input_shape: Dimensi input gambar (tinggi, lebar, channel)
        
    Returns:
        Model Keras untuk ekstraksi fitur
    """
    inputs = Input(shape=input_shape)
    
    # Layer 1: Konvolusi 64 filter 10x10
    x = Conv2D(64, (10, 10), activation='relu', name='conv2d')(inputs)
    x = MaxPooling2D()(x)
    
    # Layer 2: Konvolusi 128 filter 7x7
    x = Conv2D(128, (7, 7), activation='relu', name='conv2d_1')(x)
    x = MaxPooling2D()(x)
    
    # Layer 3: Konvolusi 128 filter 4x4
    x = Conv2D(128, (4, 4), activation='relu', name='conv2d_2')(x)
    x = MaxPooling2D()(x)
    
    # Layer 4: Konvolusi 256 filter 4x4
    x = Conv2D(256, (4, 4), activation='relu', name='conv2d_3')(x)
    x = MaxPooling2D()(x)
    
    # Flatten dan Dense layer
    x = Flatten()(x)
    x = Dense(4096, activation='sigmoid', name='dense1')(x)
    
    # Normalisasi L2
    x = Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=1))(x)
    
    model = Model(inputs=inputs, outputs=x)
    return model

def build_siamese_network():
    """
    Membangun jaringan Siamese lengkap.
    
    Returns:
        Tuple dari (model_siamese, feature_extractor)
    """
    # Build feature extractor
    feature_extractor = build_feature_extractor()
    
    # Create siamese network
    input_a = Input(shape=(128, 128, 1), name='input_layer_1')
    input_b = Input(shape=(128, 128, 1), name='input_layer_2')
    
    # Share weights between twin networks
    encoded_a = feature_extractor(input_a)
    encoded_b = feature_extractor(input_b)
    
    # Add L2 distance between the embeddings
    l2_distance = Lambda(euclidean_distance, name='lambda')([encoded_a, encoded_b])
    
    # Create the siamese network
    model = Model(inputs=[input_a, input_b], outputs=l2_distance, name='functional')
    
    # Compile model
    model.compile(loss=contrastive_loss, optimizer='adam', metrics=['accuracy'])
    
    return model, feature_extractor

class PalmPrintRecognizerV2:
    """
    Kelas utama untuk pengenalan telapak tangan.
    
    Kelas ini menyediakan fungsi-fungsi untuk:
    1. Preprocessing gambar telapak tangan
    2. Ekstraksi fitur menggunakan CNN
    3. Perbandingan kemiripan antara dua gambar
    4. Verifikasi identitas
    5. Identifikasi telapak tangan dari database
    
    Attributes:
        model: Model Siamese lengkap
        feature_extractor: Model CNN untuk ekstraksi fitur
        preprocessor: Instance dari PalmPreprocessor
        embedding_db: Database embedding telapak tangan
    """
    
    def __init__(self, model_json_path: str, model_weights_path: str):
        """
        Inisialisasi pengenal telapak tangan.
        
        Args:
            model_json_path: Path ke file JSON model
            model_weights_path: Path ke file bobot model
            
        Raises:
            Exception: Jika terjadi kesalahan saat memuat model atau inisialisasi
        """
        try:
            # Bangun model dan feature extractor
            self.model, self.feature_extractor = build_siamese_network()
            
            # Muat bobot jika file ada
            if os.path.exists(model_weights_path):
                try:
                    self.model.load_weights(model_weights_path)
                    logger.info("Model weights loaded successfully")
                except Exception as e:
                    logger.error(f"Error loading model weights: {e}")
                    raise
            else:
                logger.warning(f"Weights file not found: {model_weights_path}")
            
            # Inisialisasi preprocessor
            self.preprocessor = PalmPreprocessor()
            logger.info("Palm preprocessor initialized")
            
            # Inisialisasi database embedding
            self.embedding_db: Dict[str, np.ndarray] = {}
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
            
    def preprocess_image(self, image):
        """
        Melakukan preprocessing pada gambar telapak tangan.
        
        Args:
            image: Bisa berupa path file (str) atau array numpy
            
        Returns:
            Array numpy hasil preprocessing
            
        Raises:
            ValueError: Jika gambar tidak valid atau preprocessing gagal
        """
        return self.preprocessor.preprocess_image(image)
        
    def get_embedding(self, image):
        """
        Mendapatkan vektor embedding dari gambar telapak tangan.
        
        Args:
            image: Bisa berupa path file atau array hasil preprocessing
            
        Returns:
            Vektor embedding sebagai array numpy
            
        Raises:
            ValueError: Jika ekstraksi fitur gagal
        """
        if isinstance(image, str):
            image = self.preprocess_image(image)
        
        # Pastikan gambar memiliki dimensi yang benar
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)
        
        return self.feature_extractor.predict(image)[0]
        
    def compare_images(self, image1, image2):
        """
        Membandingkan dua gambar telapak tangan dan mengembalikan skor kemiripan.
        
        Args:
            image1: Gambar pertama (array numpy atau path)
            image2: Gambar kedua (array numpy atau path)
            
        Returns:
            Skor kemiripan antara 0 dan 1
            
        Raises:
            ValueError: Jika gambar tidak valid atau perbandingan gagal
        """
        # Dapatkan embedding untuk kedua gambar
        embedding1 = self.get_embedding(image1)
        embedding2 = self.get_embedding(image2)
        
        # Hitung jarak Euclidean
        distance = np.linalg.norm(embedding1 - embedding2)
        
        # Konversi jarak ke similarity score
        similarity = np.exp(-distance)
        return float(similarity)
        
    def verify_palm(self, image1, image2, threshold=0.7):
        """
        Verifikasi apakah dua gambar telapak tangan berasal dari orang yang sama.
        
        Args:
            image1: Gambar telapak tangan pertama
            image2: Gambar telapak tangan kedua
            threshold: Ambang batas kemiripan (default: 0.7)
            
        Returns:
            Tuple dari (is_match: bool, similarity: float)
            
        Raises:
            ValueError: Jika verifikasi gagal
        """
        similarity = self.compare_images(image1, image2)
        return similarity >= threshold, similarity
        
    def add_to_database(self, person_id: str, image: np.ndarray):
        """
        Menambahkan gambar telapak tangan ke database.
        
        Args:
            person_id: ID unik untuk orang tersebut
            image: Gambar telapak tangan
            
        Raises:
            ValueError: Jika gambar tidak valid atau penambahan gagal
        """
        embedding = self.get_embedding(image)
        self.embedding_db[person_id] = embedding
        
    def identify_palm(self, query_image: np.ndarray, threshold=0.7):
        """
        Mengidentifikasi orang dari gambar telapak tangan.
        
        Args:
            query_image: Gambar telapak tangan yang akan diidentifikasi
            threshold: Ambang batas kemiripan (default: 0.7)
            
        Returns:
            Tuple dari (person_id: str, similarity: float)
            Mengembalikan ("unknown", 0.0) jika tidak ada yang cocok
            
        Raises:
            ValueError: Jika identifikasi gagal
        """
        if not self.embedding_db:
            return "unknown", 0.0
            
        query_embedding = self.get_embedding(query_image)
        
        best_match = "unknown"
        best_similarity = 0.0
        
        for person_id, stored_embedding in self.embedding_db.items():
            distance = np.linalg.norm(query_embedding - stored_embedding)
            similarity = np.exp(-distance)
            
            if similarity > best_similarity and similarity >= threshold:
                best_match = person_id
                best_similarity = similarity
                
        return best_match, float(best_similarity)
        
    def save_database(self, db_path: str):
        """
        Menyimpan database embedding ke file.
        
        Args:
            db_path: Path untuk menyimpan file JSON database
            
        Raises:
            IOError: Jika penyimpanan gagal
        """
        # Konversi numpy arrays ke list
        serializable_db = {
            k: v.tolist() for k, v in self.embedding_db.items()
        }
        
        with open(db_path, 'w') as f:
            json.dump(serializable_db, f)
            
    def load_database(self, db_path: str):
        """
        Memuat database embedding dari file.
        
        Args:
            db_path: Path ke file JSON database
            
        Raises:
            IOError: Jika pemuatan gagal
        """
        with open(db_path, 'r') as f:
            loaded_db = json.load(f)
            
        # Konversi list kembali ke numpy arrays
        self.embedding_db = {
            k: np.array(v) for k, v in loaded_db.items()
        }

# Example usage
if __name__ == "__main__":
    # Initialize recognizer
    MODEL_JSON_PATH = "src/models/palm_print_siamese_model_v3.json"
    MODEL_WEIGHTS_PATH = "src/models/palm_print_siamese_model_v3.h5"
    
    recognizer = PalmPrintRecognizerV2(MODEL_JSON_PATH, MODEL_WEIGHTS_PATH)
    
    # Test verification
    test_image1 = "path/to/test/image1.jpg"
    test_image2 = "path/to/test/image2.jpg"
    
    if os.path.exists(test_image1) and os.path.exists(test_image2):
        is_match, similarity = recognizer.verify_palm(test_image1, test_image2)
        print(f"Verification result: {'Match' if is_match else 'No match'}")
        print(f"Similarity score: {similarity:.4f}")
    
    # Test identification
    test_db_path = "path/to/test/database.json"
    if os.path.exists(test_db_path):
        recognizer.load_database(test_db_path)
        best_match, similarity = recognizer.identify_palm(test_image1)
        print(f"Best match: {best_match}")
        print(f"Similarity score: {similarity:.4f}")
