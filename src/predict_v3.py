import os
import cv2
import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, Union
from preprocessing.palm_processorV5 import PalmPreprocessor
import logging
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Lambda

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def euclidean_distance(vects):
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
    
    # Compile model with contrastive loss
    model.compile(loss=contrastive_loss, optimizer='adam', metrics=['accuracy'])
    
    return model, feature_extractor

class PalmPrintRecognizerV3:
    def __init__(self, model_json_path: str, model_weights_path: str):
        try:
            print(f"\nLoading model from:\nJSON: {model_json_path}\nWeights: {model_weights_path}\n")
            
            # Build model and feature extractor
            self.model, self.feature_extractor = build_siamese_network()
            
            # Load weights if file exists
            if os.path.exists(model_weights_path):
                try:
                    self.model.load_weights(model_weights_path)
                    logger.info("Model weights loaded successfully")
                except Exception as e:
                    logger.error(f"Error loading model weights: {e}")
                    raise
            else:
                logger.warning(f"Weights file not found: {model_weights_path}")
            
            # Initialize preprocessor
            self.preprocessor = PalmPreprocessor()
            logger.info("Palm preprocessor initialized")
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
            
    def _preprocess_image(self, image):
        """Preprocess a palm print image."""
        try:
            return self.preprocessor.preprocess_image(image)
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            raise
            
    def get_embedding(self, image):
        """Get embedding vector for a palm print image."""
        if isinstance(image, str):
            image = self._preprocess_image(image)
            
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)
        
        return self.feature_extractor.predict(image)[0]
        
    def compare_images(self, image1, image2):
        """Compare two palm print images and return similarity score and distance."""
        # Get embeddings
        embedding1 = self.get_embedding(image1)
        embedding2 = self.get_embedding(image2)
        
        # Calculate Euclidean distance
        distance = np.linalg.norm(embedding1 - embedding2)
        
        # Convert distance to similarity score
        similarity = np.exp(-1.5 * distance)
            
        return float(similarity), float(distance)
        
    def verify_palm(self, image1, image2, base_threshold=0.65):
        """Verify if two palm print images belong to the same person."""
        try:
            # Preprocess both images
            processed_img1 = self._preprocess_image(image1)
            processed_img2 = self._preprocess_image(image2)
            
            # Compare images
            similarity, distance = self.compare_images(processed_img1, processed_img2)
            
            # Adaptive threshold with finer granularity and balanced thresholds
            if distance < 0.14:
                threshold = base_threshold * 0.9    # 0.585 - Perfect match
            elif 0.14 <= distance < 0.145:
                threshold = base_threshold * 1.15   # 0.7475 - Very similar, likely same person
            elif 0.145 <= distance < 0.15:
                threshold = base_threshold * 1.231   # 0.8001 - Moderate for potential false negatives
            elif 0.15 <= distance < 0.155:
                threshold = base_threshold * 1.25   # 0.8125 - Strict for potential false positives
            elif 0.155 <= distance < 0.16:
                threshold = base_threshold * 1.3    # 0.845 - Very strict for high distance
            elif 0.16 <= distance < 0.19:
                threshold = base_threshold * 1.16   # 0.78 - Standard different person range
            else:
                threshold = base_threshold * 1.16   # 0.75 - Clearly different
                
            # Determine match based on threshold and apply distance penalty
            # Add small penalty for larger distances to help differentiate edge cases
            distance_penalty = max(0, (distance - 0.14) * 0.1)
            adjusted_similarity = similarity - distance_penalty
            
            is_match = adjusted_similarity >= threshold
            
            logger.info(f"Verification Test Result: Match={is_match}, Similarity={similarity:.4f}, Adjusted={adjusted_similarity:.4f}, Distance={distance:.4f}, Threshold={threshold:.4f}")
            return is_match, similarity, distance
            
        except Exception as e:
            logger.error(f"Failed to preprocess one or both images")
            return False, 0.0, float('inf')
