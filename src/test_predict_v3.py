import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from predict_v3 import PalmPrintRecognizerV3
import logging
import time
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class PalmRecognitionTesterV3:
    """
    Kelas untuk testing palm print recognition dengan visualisasi hasil
    """
    def __init__(self, model_json_path: str, model_weights_path: str):
        """
        Inisialisasi tester
        Args:
            model_json_path: Path ke file JSON arsitektur model
            model_weights_path: Path ke file H5 bobot model
        """
        self.recognizer = PalmPrintRecognizerV3(model_json_path, model_weights_path)
        self.test_results = []
        
    def test_verification(self, image1, image2, expected_match: bool = None) -> dict:
        """
        Test verifikasi telapak tangan
        Args:
            image1: Gambar pertama (path atau array)
            image2: Gambar kedua (path atau array)
            expected_match: Hasil yang diharapkan (jika diketahui)
        Returns:
            Dictionary berisi hasil test
        """
        try:
            start_time = time.time()
            is_match, similarity, distance = self.recognizer.verify_palm(image1, image2)
            processing_time = time.time() - start_time
            
            result = {
                'test_type': 'verification',
                'is_match': is_match,
                'similarity': similarity,
                'distance': distance,
                'processing_time': processing_time
            }
            
            if expected_match is not None:
                result['expected_match'] = expected_match
                result['correct_prediction'] = is_match == expected_match
            
            self.test_results.append(result)
            logger.info(f"Verification Test Result: Match={is_match}, "
                       f"Similarity={similarity:.4f}, Distance={distance:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in verification test: {e}")
            raise
            
    def plot_distance_analysis(self, result: Dict[str, Any]):
        """
        Visualisasi analisis distance dan similarity
        Args:
            result: Dictionary hasil test
        """
        distance = result['distance']
        similarity = result['similarity']
        
        # Plot distance analysis
        plt.figure(figsize=(15, 5))
        
        # 1. Distance Plot
        plt.subplot(131)
        distances = np.linspace(0, 600, 1000)
        plt.plot(distances, [300] * len(distances), 'r--', label='Threshold (300)')
        plt.axvline(x=distance, color='b', linestyle='-', 
                   label=f'Current Distance ({distance:.2f})')
        plt.fill_between(distances, 0, 300, alpha=0.2, color='g', label='Match Zone')
        plt.fill_between(distances, 300, 600, alpha=0.2, color='r', label='No Match Zone')
        plt.grid(True, alpha=0.3)
        plt.title('Distance Analysis')
        plt.xlabel('L2 Distance')
        plt.ylabel('Threshold')
        plt.legend()
        
        # 2. Similarity Plot
        plt.subplot(132)
        similarities = np.linspace(0, 1, 1000)
        plt.plot(similarities, similarities, 'g--', label='Linear Reference')
        plt.axvline(x=similarity, color='b', linestyle='-',
                   label=f'Current Similarity ({similarity:.4f})')
        plt.grid(True, alpha=0.3)
        plt.title('Similarity Analysis')
        plt.xlabel('Similarity Score')
        plt.ylabel('Reference')
        plt.legend()
        
        # 3. Distance-Similarity Relationship
        plt.subplot(133)
        distances = np.linspace(0, 600, 1000)
        similarities = np.exp(-0.005 * distances)  # Same alpha as in predict_v3.py
        plt.plot(distances, similarities, 'b-', label='Distance-Similarity Curve')
        plt.axvline(x=distance, color='r', linestyle='--',
                   label=f'Current Distance ({distance:.2f})')
        plt.axhline(y=similarity, color='g', linestyle='--',
                   label=f'Current Similarity ({similarity:.4f})')
        plt.grid(True, alpha=0.3)
        plt.title('Distance-Similarity Relationship')
        plt.xlabel('L2 Distance')
        plt.ylabel('Similarity Score')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_verification_results(self, same_person_results, diff_person_results):
        """Plot verification results analysis."""
        plt.figure(figsize=(15, 6))
        
        # Plot similarity vs distance scatter
        plt.subplot(121)
        same_dist, same_sim = zip(*[(d, s) for _, s, d in same_person_results])
        diff_dist, diff_sim = zip(*[(d, s) for _, s, d in diff_person_results])
        
        plt.scatter(same_dist, same_sim, c='green', label='Same Person', alpha=0.6, s=100)
        plt.scatter(diff_dist, diff_sim, c='red', label='Different Person', alpha=0.6, s=100)
        plt.xlabel('Distance')
        plt.ylabel('Similarity')
        plt.title('Similarity vs Distance Analysis')
        
        # Add 80% threshold line
        plt.axhline(y=0.75, color='orange', linestyle='--', label='75% Threshold')
        plt.fill_between(plt.xlim(), 0.75, 1.0, color='green', alpha=0.1, label='Match Zone (>75%)')
        plt.fill_between(plt.xlim(), 0, 0.75, color='red', alpha=0.1, label='No Match Zone (<75%)')
        
        # Add text annotations for threshold
        for d, s in zip(same_dist + diff_dist, same_sim + diff_sim):
            if s >= 0.75:
                status = "Match"
                color = 'green'
            else:
                status = "No Match"
                color = 'red'
            plt.annotate(f'{s:.1%}\n({status})', 
                        (d, s), 
                        xytext=(5, 5), 
                        textcoords='offset points',
                        color=color,
                        fontsize=8)
        
        plt.legend()
        plt.grid(True)
        
        # Add threshold curve
        distances = np.linspace(0, 0.3, 100)
        thresholds = []
        base_threshold = 0.65
        
        for d in distances:
            if d < 0.14:
                t = base_threshold * 0.9
            elif 0.14 <= d < 0.145:
                t = base_threshold * 1.15
            elif 0.145 <= d < 0.15:
                t = base_threshold * 1.231
            elif 0.15 <= d < 0.155:
                t = base_threshold * 1.25
            elif 0.155 <= d < 0.16:
                t = base_threshold * 1.3
            elif 0.16 <= d < 0.19:
                t = base_threshold * 1.16
            else:
                t = base_threshold * 1.16
            thresholds.append(t)
        
        plt.plot(distances, thresholds, 'b--', label='Threshold', alpha=0.7)
        
        # Plot threshold ranges
        plt.subplot(122)
        plt.plot(distances, thresholds, 'b-', label='Threshold', linewidth=2)
        
        # Add range annotations
        plt.axvline(x=0.14, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(x=0.145, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(x=0.15, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(x=0.155, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(x=0.16, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(x=0.19, color='gray', linestyle='--', alpha=0.5)
        
        plt.text(0.07, 0.6, 'Perfect\nMatch', ha='center')
        plt.text(0.142, 0.75, 'Very\nSimilar', ha='center')
        plt.text(0.147, 0.7, 'Border\nline', ha='center')
        plt.text(0.152, 0.81, 'Strict', ha='center')
        plt.text(0.157, 0.84, 'Very\nStrict', ha='center')
        plt.text(0.175, 0.78, 'Different', ha='center')
        
        plt.xlabel('Distance')
        plt.ylabel('Threshold')
        plt.title('Adaptive Threshold Analysis')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.show()

def test_palm_recognition():
    """Test palm print recognition dengan improved visualization"""
    # Initialize model with correct paths
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    model_json = os.path.join(model_dir, 'palm_print_siamese_model_v3.json')
    model_weights = os.path.join(model_dir, 'palm_print_siamese_model_v3.h5')
    
    print(f"\nLoading model from:")
    print(f"JSON: {model_json}")
    print(f"Weights: {model_weights}")
    
    # Create tester
    tester = PalmRecognitionTesterV3(model_json, model_weights)
    
    # Test directory
    test_dir = os.path.join(os.path.dirname(__file__), '..', 'test_image')
    
    test_cases = [
        {
            "name": "Same Person (ALI) - Same Image",
            "img1": os.path.join(test_dir, "ALI", "001.jpg"),
            "img2": os.path.join(test_dir, "ALI", "001.jpg"),
            "expected_match": True
        },
        {
            "name": "Same Person (ALI) - Different Images",
            "img1": os.path.join(test_dir, "ALI", "001.jpg"),
            "img2": os.path.join(test_dir, "ALI", "005.jpg"),
            "expected_match": True
        },
        {
            "name": "Different Person (ALI vs ARA)",
            "img1": os.path.join(test_dir, "ALI", "001.jpg"),
            "img2": os.path.join(test_dir, "ARA", "001.jpg"),
            "expected_match": False
        },
        {
            "name": "Different Person (ALI vs ARA) - Different Images",
            "img1": os.path.join(test_dir, "ALI", "002.jpg"),
            "img2": os.path.join(test_dir, "ARA", "002.jpg"),
            "expected_match": False
        }
    ]

    # Run test cases
    same_person_results = []
    diff_person_results = []
    
    for test_case in test_cases:
        print(f"\n=== {test_case['name']} ===")
        print("-" * 50)
        print("Testing images:")
        print(f"Image 1: {test_case['img1']}")
        print(f"Image 2: {test_case['img2']}\n")

        result = tester.test_verification(test_case['img1'], test_case['img2'], test_case['expected_match'])
        tester.plot_distance_analysis(result)

        if test_case['expected_match']:
            same_person_results.append((result['is_match'], result['similarity'], result['distance']))
        else:
            diff_person_results.append((result['is_match'], result['similarity'], result['distance']))

        result_str = "PASS" if result['correct_prediction'] else "FAIL"
        print(f"Test Result: {result_str}")
        print(f"Expected Match: {test_case['expected_match']}")
        print(f"Actual Match: {result['is_match']}")
        print(f"Similarity Score: {result['similarity']:.4f}")
        print(f"Distance: {result['distance']:.4f}")

    tester.plot_verification_results(same_person_results, diff_person_results)

if __name__ == "__main__":
    test_palm_recognition()
