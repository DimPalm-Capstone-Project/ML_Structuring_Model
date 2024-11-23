import os
import numpy as np
from predict import PalmPrintRecognizer
from pathlib import Path

class TrainedImageTester:
    def __init__(self, model_path='src/models/palm_print_siamese_model.h5'):
        self.recognizer = PalmPrintRecognizer(model_path)
        self.data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
        
    def build_database(self):
        """Build database from training images"""
        print("\nBuilding database from training images...")
        image_files = sorted([f for f in os.listdir(self.data_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        # Add all images to database
        for img_file in image_files:
            person_id = os.path.splitext(img_file)[0]  # DATASET-001
            img_path = os.path.join(self.data_dir, img_file)
            self.recognizer.add_to_database(person_id, img_path)
            print(f"Added {img_file} to database")
            
        return image_files  # Return all images for testing
    
    def test_recognition(self, test_images, threshold=0.98):
        """Test recognition on the given test images"""
        print(f"\nTesting recognition on training images (threshold: {threshold})...")
        results = []
        
        for img_file in test_images:
            true_id = os.path.splitext(img_file)[0]  # DATASET-001
            img_path = os.path.join(self.data_dir, img_file)
            
            # Get predictions
            best_match, similarity = self.recognizer.find_match(img_path, threshold=threshold)
            
            # Store results
            match_correct = (true_id == best_match) if best_match else False
            result = {
                'image': img_file,
                'true_id': true_id,
                'predicted_id': best_match if best_match else "No match",
                'similarity': similarity,
                'correct': match_correct
            }
            results.append(result)
            
            # Print result
            print(f"\nTesting {img_file}:")
            print(f"True ID: {true_id}")
            print(f"Predicted ID: {result['predicted_id']}")
            print(f"Similarity Score: {similarity:.4f}")
            print(f"Correct: {'YES' if match_correct else 'NO'}")
        
        # Calculate metrics
        total = len(results)
        correct = sum(1 for r in results if r['correct'])
        no_match = sum(1 for r in results if r['predicted_id'] == "No match")
        wrong_match = total - correct - no_match
        
        print(f"\nResults Summary:")
        print(f"Total Images Tested: {total}")
        print(f"Correct Matches: {correct}")
        print(f"No Matches Found: {no_match}")
        print(f"Wrong Matches: {wrong_match}")
        print(f"Accuracy: {(correct/total):.2%}")
        
        return results

def main():
    print("Starting Palm Print Recognition Testing on Training Images")
    print("=" * 50)
    
    # Initialize tester
    tester = TrainedImageTester()
    
    # Build database and get test images
    test_images = tester.build_database()
    
    # Test with different thresholds
    thresholds = [0.95, 0.98, 0.99]
    for threshold in thresholds:
        results = tester.test_recognition(test_images, threshold)
        print("\n" + "="*50)
    
    print("\nTesting Complete!")

if __name__ == "__main__":
    main()
