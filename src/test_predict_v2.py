import os
import logging
import numpy as np
from predict_v2 import PalmPrintRecognizerV2
import matplotlib.pyplot as plt
from typing import List, Tuple
import time
import cv2

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PalmRecognitionTester:
    def __init__(self, model_json_path: str, model_weights_path: str):
        """Initialize the tester with model paths"""
        self.recognizer = PalmPrintRecognizerV2(model_json_path, model_weights_path)
        self.test_results = []
        
    def test_verification(self, image1, image2, expected_match: bool = None) -> dict:
        """
        Test palm verification between two images
        
        Args:
            image1: First image (path or preprocessed numpy array)
            image2: Second image (path or preprocessed numpy array)
            expected_match: Expected match result (if known)
            
        Returns:
            Dictionary containing test results
        """
        try:
            start_time = time.time()
            is_match, similarity = self.recognizer.verify_palm(image1, image2)
            processing_time = time.time() - start_time
            
            result = {
                'test_type': 'verification',
                'is_match': is_match,
                'similarity': similarity,
                'processing_time': processing_time
            }
            
            if expected_match is not None:
                result['expected_match'] = expected_match
                result['correct_prediction'] = is_match == expected_match
            
            self.test_results.append(result)
            logger.info(f"Verification Test Result: Match={is_match}, Similarity={similarity:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in verification test: {e}")
            raise

    def test_identification(self, test_image_path: str, database_paths: dict) -> dict:
        """
        Test palm identification against a database
        
        Args:
            test_image_path: Path to test image
            database_paths: Dictionary of {person_id: image_path}
            
        Returns:
            Dictionary containing test results
        """
        try:
            # Clear existing database
            self.recognizer.embedding_db = {}
            
            # Add images to database
            for person_id, image_path in database_paths.items():
                self.recognizer.add_to_database(person_id, image_path)
            
            start_time = time.time()
            best_match, similarity = self.recognizer.identify_palm(test_image_path)
            processing_time = time.time() - start_time
            
            result = {
                'test_type': 'identification',
                'test_image': test_image_path,
                'database_size': len(database_paths),
                'best_match': best_match,
                'similarity': similarity,
                'processing_time': processing_time
            }
            
            self.test_results.append(result)
            logger.info(f"Identification Test Result: Best Match={best_match}, Similarity={similarity:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in identification test: {e}")
            raise

    def test_threshold_sensitivity(self, image_pairs: List[Tuple[str, str]], 
                                 thresholds: List[float]) -> dict:
        """
        Test model performance across different similarity thresholds
        
        Args:
            image_pairs: List of tuples containing paths to image pairs
            thresholds: List of threshold values to test
            
        Returns:
            Dictionary containing sensitivity analysis results
        """
        try:
            results = {}
            for threshold in thresholds:
                matches = []
                similarities = []
                
                for img1_path, img2_path in image_pairs:
                    is_match, similarity = self.recognizer.verify_palm(img1_path, img2_path)
                    matches.append(is_match)
                    similarities.append(similarity)
                
                results[threshold] = {
                    'matches': matches,
                    'similarities': similarities,
                    'match_rate': sum(matches) / len(matches)
                }
                
            logger.info("Completed threshold sensitivity analysis")
            return results
            
        except Exception as e:
            logger.error(f"Error in threshold sensitivity test: {e}")
            raise

    def plot_results(self):
        """Plot test results"""
        try:
            verification_results = [r for r in self.test_results if r['test_type'] == 'verification']
            identification_results = [r for r in self.test_results if r['test_type'] == 'identification']
            
            plt.figure(figsize=(15, 5))
            
            # Plot verification similarities
            if verification_results:
                plt.subplot(131)
                similarities = [r['similarity'] for r in verification_results]
                plt.hist(similarities, bins=10, alpha=0.7)
                plt.title('Verification Similarities Distribution')
                plt.xlabel('Similarity Score')
                plt.ylabel('Count')
            
            # Plot processing times
            if self.test_results:
                plt.subplot(132)
                times = [r['processing_time'] for r in self.test_results]
                plt.hist(times, bins=10, alpha=0.7)
                plt.title('Processing Time Distribution')
                plt.xlabel('Time (seconds)')
                plt.ylabel('Count')
            
            # Plot identification results
            if identification_results:
                plt.subplot(133)
                similarities = [r['similarity'] for r in identification_results]
                plt.hist(similarities, bins=10, alpha=0.7)
                plt.title('Identification Similarities Distribution')
                plt.xlabel('Similarity Score')
                plt.ylabel('Count')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting results: {e}")
            raise

def test_palm_recognition():
    """Test palm print recognition with improved visualization"""
    # Initialize model with correct paths
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    model_json = os.path.join(model_dir, 'palm_print_siamese_model_v3.json')
    model_weights = os.path.join(model_dir, 'palm_print_siamese_model_v3.h5')
    
    print(f"\nLoading model from:")
    print(f"JSON: {model_json}")
    print(f"Weights: {model_weights}")
    
    # Create tester
    tester = PalmRecognitionTester(model_json, model_weights)
    
    # Load test images
    test_dir = os.path.join(os.path.dirname(__file__), '..', 'test_image')
    img1_path = os.path.join(test_dir, 'ALI', '001.jpg')  # ALI's palm
    img2_path = os.path.join(test_dir, 'ARA', '001.jpg')  # ARA's palm (different image)
    
    # Test different persons (ALI vs ARA)
    print("\nTesting similarity between ALI and ARA's palm prints:")
    print("-" * 50)
    
    # Load and preprocess images once
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        raise ValueError("Failed to load test images")
    
    # Preprocess images once
    processed1 = tester.recognizer.preprocessor.preprocess_image(img1)
    processed2 = tester.recognizer.preprocessor.preprocess_image(img2)
    
    if processed1 is None or processed2 is None:
        print("\nPreprocessing failed for one or both images:")
        if processed1 is None:
            print("- ALI's image failed preprocessing")
        if processed2 is None:
            print("- FIRA's image failed preprocessing")
        return
    
    # Test verification with preprocessed images
    logger.info("Testing verification with different persons...")
    result = tester.test_verification(processed1, processed2, expected_match=False)
    
    # Test threshold sensitivity
    logger.info("Testing threshold sensitivity...")
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    similarity = result["similarity"]
    
    # Plot threshold analysis
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.axhline(y=similarity, color='r', linestyle='--', label=f'Current Similarity ({similarity:.3f})')
    plt.plot(thresholds, thresholds, 'b-', label='Threshold Line')
    plt.fill_between(thresholds, thresholds, 1, alpha=0.2, color='g', label='Match Zone')
    plt.fill_between(thresholds, 0, thresholds, alpha=0.2, color='r', label='No Match Zone')
    plt.grid(True)
    plt.xlabel('Threshold')
    plt.ylabel('Required Similarity')
    plt.title('Threshold Analysis')
    plt.legend()
    
    # Add match/no-match status for each threshold
    for t in thresholds:
        status = "Match" if similarity >= t else "No Match"
        plt.text(t, similarity + 0.02, f'T={t:.1f}\n{status}', 
                horizontalalignment='center', verticalalignment='bottom')
    
    # Show difference map
    plt.subplot(122)
    diff = np.abs(processed1 - processed2)
    plt.imshow(diff, cmap='hot')
    plt.title('Difference Map')
    plt.colorbar()
    plt.axis('off')
    
    plt.suptitle(f'Similarity Analysis (Score = {similarity:.4f})')
    plt.tight_layout()
    plt.show()
    
    # Show original and processed images
    plt.figure(figsize=(10, 5))
    
    # Original images
    plt.subplot(221)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title('ALI 001.jpg')
    plt.axis('off')
    
    plt.subplot(222)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title('ARA 002.jpg')
    plt.axis('off')
    
    # Preprocessed images
    plt.subplot(223)
    plt.imshow(processed1, cmap='gray')
    plt.title('Processed ALI')
    plt.axis('off')
    
    plt.subplot(224)
    plt.imshow(processed2, cmap='gray')
    plt.title('Processed ARA')
    plt.axis('off')
    
    plt.suptitle('Original vs Processed Images')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_palm_recognition()
