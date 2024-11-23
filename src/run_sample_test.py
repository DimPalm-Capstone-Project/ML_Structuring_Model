import os
from add_and_test_new_data import NewDataTester

def run_sample_test():
    # Initialize the tester
    tester = NewDataTester()
    
    # Get paths
    test_samples_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'test_samples')
    
    # First, add some images to the database
    print("\n1. Adding training images to database...")
    training_images = [
        ('person1_sample_1.png', 'person1'),
        ('person1_sample_2.png', 'person1'),
        ('person2_sample_1.png', 'person2'),
        ('person2_sample_2.png', 'person2')
    ]
    
    for img_name, person_name in training_images:
        img_path = os.path.join(test_samples_dir, img_name)
        if os.path.exists(img_path):
            tester.add_new_image(img_path, person_name)
    
    # Now test recognition with the remaining images
    print("\n2. Testing recognition with new images...")
    test_images = [
        'person1_sample_3.png',  # Should match person1
        'person2_sample_3.png'   # Should match person2
    ]
    
    for test_img in test_images:
        print(f"\nTesting image: {test_img}")
        test_path = os.path.join(test_samples_dir, test_img)
        if os.path.exists(test_path):
            results = tester.test_new_image(test_path, threshold=0.5)
            
            print("\nTop matches:")
            print("-" * 50)
            for i, result in enumerate(results[:3], 1):
                print(f"{i}. Person: {result['person_name']}")
                print(f"   Similarity: {result['similarity']:.4f}")
                print(f"   Match: {'Yes' if result['match'] else 'No'}")
                print("-" * 50)
        else:
            print(f"Error: Test image not found: {test_path}")

if __name__ == "__main__":
    run_sample_test()
