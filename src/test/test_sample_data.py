import os
import numpy as np
import cv2

def create_sample_images():
    # Create test directory
    test_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'test_samples')
    os.makedirs(test_dir, exist_ok=True)
    
    # Create synthetic palm print-like images for testing
    image_size = (128, 128)
    num_samples = 3
    
    # Create samples for two different people
    for person_id in ['person1', 'person2']:
        # Create base pattern for this person
        base_pattern = np.random.rand(*image_size)
        
        for i in range(num_samples):
            # Add some random variation to the base pattern
            noise = np.random.normal(0, 0.1, image_size)
            image = base_pattern + noise
            
            # Normalize to 0-255 range
            image = ((image - image.min()) * (255.0 / (image.max() - image.min()))).astype(np.uint8)
            
            # Add some line patterns to simulate palm lines
            for _ in range(5):
                pt1 = (np.random.randint(0, 128), np.random.randint(0, 128))
                pt2 = (np.random.randint(0, 128), np.random.randint(0, 128))
                cv2.line(image, pt1, pt2, (200, 200, 200), 1)
            
            # Save the image
            filename = f"{person_id}_sample_{i+1}.png"
            filepath = os.path.join(test_dir, filename)
            cv2.imwrite(filepath, image)
            print(f"Created sample image: {filename}")
    
    return test_dir

if __name__ == "__main__":
    sample_dir = create_sample_images()
    print(f"\nSample images created in: {sample_dir}")
