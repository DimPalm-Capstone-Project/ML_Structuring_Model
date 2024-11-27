import os
import shutil
import numpy as np
import cv2
from predict import PalmPrintRecognizer
from datetime import datetime

class NewDataTester:
    def __init__(self, model_weights_path='src/models/palm_print_siamese_model.h5'):
        self.recognizer = PalmPrintRecognizer(model_weights_path)
        self.new_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'new_palm_prints')
        os.makedirs(self.new_data_dir, exist_ok=True)
    
    def add_new_image(self, image_path, person_name):
        """
        Add a new palm print image to the database
        """
        # Create person directory if it doesn't exist
        person_dir = os.path.join(self.new_data_dir, person_name)
        os.makedirs(person_dir, exist_ok=True)
        
        # Copy image to person directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_ext = os.path.splitext(image_path)[1]
        new_image_name = f"{person_name}_{timestamp}{image_ext}"
        new_image_path = os.path.join(person_dir, new_image_name)
        
        shutil.copy2(image_path, new_image_path)
        print(f"Added new image for {person_name}: {new_image_name}")
        
        return new_image_path

    def preprocess_image(self, image_path):
        """
        Preprocess a single image for recognition
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))
        img = img.astype('float32') / 255.0
        return np.expand_dims(img, axis=-1)

    def test_new_image(self, test_image_path, threshold=0.5):
        """
        Test recognition of a new image against the existing database
        """
        # Preprocess test image
        test_img = self.preprocess_image(test_image_path)
        test_img = np.expand_dims(test_img, axis=0)
        
        # Get embedding for test image
        test_embedding = self.recognizer.get_embedding(test_img)
        
        # Compare with all images in the database
        results = []
        for person_name in os.listdir(self.new_data_dir):
            person_dir = os.path.join(self.new_data_dir, person_name)
            if os.path.isdir(person_dir):
                for img_name in os.listdir(person_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(person_dir, img_name)
                        db_img = self.preprocess_image(img_path)
                        db_img = np.expand_dims(db_img, axis=0)
                        
                        # Get embedding and compute similarity
                        db_embedding = self.recognizer.get_embedding(db_img)
                        similarity = self.recognizer.compute_similarity(test_embedding, db_embedding)
                        
                        results.append({
                            'person_name': person_name,
                            'image_name': img_name,
                            'similarity': float(similarity),
                            'match': similarity >= threshold
                        })
        
        # Sort results by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results

def main():
    # Example usage
    tester = NewDataTester()
    
    while True:
        print("\nPalm Print Recognition - New Data Testing")
        print("1. Add new palm print image")
        print("2. Test recognition on image")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            image_path = input("Enter the path to the palm print image: ")
            person_name = input("Enter the person's name: ")
            
            if os.path.exists(image_path):
                new_path = tester.add_new_image(image_path, person_name)
                print(f"Successfully added image for {person_name}")
            else:
                print("Error: Image file not found!")
                
        elif choice == '2':
            test_image_path = input("Enter the path to the test image: ")
            if os.path.exists(test_image_path):
                threshold = float(input("Enter similarity threshold (0-1, default 0.5): ") or 0.5)
                results = tester.test_new_image(test_image_path, threshold)
                
                print("\nRecognition Results:")
                print("-" * 50)
                for i, result in enumerate(results[:5], 1):  # Show top 5 matches
                    print(f"{i}. Person: {result['person_name']}")
                    print(f"   Image: {result['image_name']}")
                    print(f"   Similarity: {result['similarity']:.4f}")
                    print(f"   Match: {'Yes' if result['match'] else 'No'}")
                    print("-" * 50)
            else:
                print("Error: Test image file not found!")
                
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice! Please try again.")

if __name__ == "__main__":
    main()
