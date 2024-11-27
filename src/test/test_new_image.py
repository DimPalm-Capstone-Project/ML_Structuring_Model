import os
import numpy as np
from predict import PalmPrintRecognizer
from pathlib import Path

class NewImageTester:
    def __init__(self, model_path='src/models/palm_print_siamese_model.h5'):
        self.recognizer = PalmPrintRecognizer(model_path)
        self.base_dir = os.path.dirname(os.path.dirname(__file__))
        self.train_dir = os.path.join(self.base_dir, 'data', 'raw')
        self.test_dir = os.path.join(self.base_dir, 'data', 'test')
        
    def build_database(self):
        """Build database from training images"""
        print("\nBuilding database from training images...")
        image_files = sorted([f for f in os.listdir(self.train_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        for img_file in image_files:
            person_id = os.path.splitext(img_file)[0]
            img_path = os.path.join(self.train_dir, img_file)
            self.recognizer.add_to_database(person_id, img_path)
            print(f"Added {img_file} to database")
    
    def test_images(self, thresholds=[0.85, 0.90, 0.95]):
        """Test all images in test directory against the database"""
        if not os.path.exists(self.test_dir):
            print(f"Test directory not found: {self.test_dir}")
            return
            
        test_images = sorted([f for f in os.listdir(self.test_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        if not test_images:
            print("No test images found!")
            return
            
        print(f"\nTesting {len(test_images)} images from: {self.test_dir}")
        print("=" * 50)
        
        results = []
        for img_file in test_images:
            img_path = os.path.join(self.test_dir, img_file)
            print(f"\nTesting image: {img_file}")
            print("-" * 30)
            
            # Get embedding for test image once
            test_embedding = self.recognizer.get_embedding(img_path)
            
            for threshold in thresholds:
                print(f"\nThreshold: {threshold}")
                print("-" * 20)
                
                # Get predictions
                best_match, similarity = self.recognizer.find_match(img_path, threshold=threshold)
                
                # Store results
                results.append({
                    'test_image': img_file,
                    'threshold': threshold,
                    'best_match': best_match,
                    'similarity': similarity
                })
                
                # Print results
                print(f"Best Match: {best_match if best_match else 'No match found'}")
                print(f"Similarity Score: {similarity:.4f}")
                
                # Get all matches sorted by similarity
                print("\nAll Matches (sorted by similarity):")
                matches = []
                for id, embedding in self.recognizer.embedding_db.items():
                    sim = self.recognizer.compute_similarity(test_embedding, embedding)
                    matches.append((id, sim))
                
                # Sort by similarity and print top matches
                matches.sort(key=lambda x: x[1], reverse=True)
                for idx, (match_id, sim) in enumerate(matches[:5], 1):  # Show top 5
                    print(f"{idx}. {match_id}: {sim:.4f}")
            
        # Print summary
        print("\nTest Summary")
        print("=" * 50)
        for threshold in thresholds:
            threshold_results = [r for r in results if r['threshold'] == threshold]
            matches = sum(1 for r in threshold_results if r['best_match'] is not None)
            print(f"\nThreshold {threshold}:")
            print(f"Total images tested: {len(threshold_results)}")
            print(f"Images matched: {matches}")
            print(f"Match rate: {matches/len(threshold_results)*100:.2f}%")

def main():
    print("Palm Print Recognition - Multiple Test Images")
    print("=" * 50)
    
    # Initialize tester
    tester = NewImageTester()
    
    # Build database
    tester.build_database()
    
    # Test all images in test directory
    tester.test_images()

if __name__ == "__main__":
    main()
