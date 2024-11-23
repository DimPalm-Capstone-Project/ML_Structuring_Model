import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Lambda
import cv2
import os
import json
from typing import Dict, List, Tuple

def build_base_network(input_shape=(128, 128, 1)):
    """Build the base network for feature extraction"""
    inputs = Input(shape=input_shape)
    
    # First block
    x = Conv2D(64, (10, 10), activation='relu')(inputs)
    x = MaxPooling2D()(x)
    x = Conv2D(128, (7, 7), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(128, (4, 4), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(256, (4, 4), activation='relu')(x)
    
    # Flatten and Dense layers
    x = Flatten()(x)
    x = Dense(4096, activation='sigmoid')(x)
    
    return Model(inputs, x)

class PalmPrintRecognizer:
    def __init__(self, model_path: str):
        # Build base network
        base_network = build_base_network()
        
        # Create Siamese network
        input_a = Input(shape=(128, 128, 1))
        input_b = Input(shape=(128, 128, 1))
        
        # Get embeddings
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)
        
        # Add Lambda layer to compute L1 distance
        L1_layer = Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))
        L1_distance = L1_layer([processed_a, processed_b])
        
        # Add Dense layer with sigmoid activation
        prediction = Dense(1, activation='sigmoid')(L1_distance)
        
        # Create model
        self.model = Model(inputs=[input_a, input_b], outputs=prediction)
        
        # Load weights
        self.model.load_weights(model_path)
        
        # Create embedding model
        self.embedding_model = Model(inputs=input_a, outputs=processed_a)
        
        # Initialize embedding database
        self.embedding_db: Dict[str, np.ndarray] = {}
        
    def preprocess_image(self, image):
        """
        Preprocess an image for the model. Can accept either a file path or a numpy array.
        """
        if isinstance(image, str):
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not load image from path: {image}")
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                img = image
        else:
            raise ValueError("Image must be either a file path or numpy array")
            
        img = cv2.resize(img, (128, 128))
        img = img.astype('float32') / 255.0
        return np.expand_dims(img, axis=-1)

    def get_embedding(self, image):
        """
        Get embedding for an image. Can accept either a file path or preprocessed image array.
        """
        if isinstance(image, str):
            preprocessed_img = self.preprocess_image(image)
            preprocessed_img = np.expand_dims(preprocessed_img, axis=0)
        else:
            preprocessed_img = image
            
        return self.embedding_model.predict(preprocessed_img)[0]  # Return flattened embedding
        
    def add_to_database(self, person_id: str, image):
        """Add a new person's palm print to database"""
        embedding = self.get_embedding(image)
        self.embedding_db[person_id] = embedding
        
    def save_database(self, db_path: str) -> None:
        """Save embedding database to file"""
        # Convert numpy arrays to lists for JSON serialization
        db_serializable = {k: v.tolist() for k, v in self.embedding_db.items()}
        with open(db_path, 'w') as f:
            json.dump(db_serializable, f)
            
    def load_database(self, db_path: str) -> None:
        """Load embedding database from file"""
        with open(db_path, 'r') as f:
            db_dict = json.load(f)
        # Convert lists back to numpy arrays
        self.embedding_db = {k: np.array(v) for k, v in db_dict.items()}
        
    def compute_similarity(self, embedding1, embedding2):
        """
        Compute similarity between two embeddings using L1 distance
        """
        # Convert to numpy arrays if they aren't already
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)
        
        # Compute L1 distance
        l1_distance = np.sum(np.abs(emb1 - emb2))
        
        # Convert distance to similarity score (0 to 1)
        similarity = 1 / (1 + l1_distance)
        
        return similarity

    def find_match(self, image, threshold: float = 0.5) -> Tuple[str, float]:
        """Find matching person in database"""
        query_embedding = self.get_embedding(image)
        
        best_match = None
        best_similarity = -1
        
        for person_id, stored_embedding in self.embedding_db.items():
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, stored_embedding) / \
                        (np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding))
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = person_id
                
        if best_similarity < threshold:
            return None, best_similarity
            
        return best_match, best_similarity

# Example usage
if __name__ == "__main__":
    # Initialize recognizer
    recognizer = PalmPrintRecognizer(
        model_path="models/palm_print_siamese_model.h5"
    )
    
    # Add some examples to database
    recognizer.add_to_database("person1", "data/person1_palm.jpg")
    recognizer.add_to_database("person2", "data/person2_palm.jpg")
    
    # Save database
    recognizer.save_database("palm_print_db.json")
    
    # Find match for new image
    person_id, similarity = recognizer.find_match("data/test_palm.jpg")
    if person_id:
        print(f"Match found: {person_id} with similarity {similarity:.2f}")
    else:
        print(f"No match found (similarity: {similarity:.2f})")
