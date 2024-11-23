import os
import random
from predict import PalmPrintRecognizer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def test_recognition_system(data_dir: str, test_split: float = 0.2):
    """
    Test the palm print recognition system
    Args:
        data_dir: Directory containing person folders with palm print images
        test_split: Proportion of images to use for testing
    """
    # Initialize recognizer
    recognizer = PalmPrintRecognizer(
        model_path="src/models/palm_print_siamese_model.h5"
    )
    
    # Collect all person directories
    person_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    # Dictionary to store train and test splits
    train_images = {}
    test_images = {}
    
    # Split images for each person
    for person_id in person_dirs:
        person_path = os.path.join(data_dir, person_id)
        images = [f for f in os.listdir(person_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        # Randomly split images
        n_test = max(1, int(len(images) * test_split))
        test_set = random.sample(images, n_test)
        train_set = [img for img in images if img not in test_set]
        
        train_images[person_id] = [os.path.join(person_path, img) for img in train_set]
        test_images[person_id] = [os.path.join(person_path, img) for img in test_set]
    
    # Build database with training images
    print("Building database...")
    for person_id, image_paths in train_images.items():
        # Use first image for each person
        recognizer.add_to_database(person_id, image_paths[0])
    
    # Test recognition
    print("\nTesting recognition...")
    true_labels = []
    pred_labels = []
    similarities = []
    
    for true_id, test_paths in test_images.items():
        for test_path in test_paths:
            pred_id, similarity = recognizer.find_match(test_path)
            true_labels.append(true_id)
            pred_labels.append(pred_id if pred_id else "unknown")
            similarities.append(similarity)
    
    # Calculate metrics
    correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
    total = len(true_labels)
    accuracy = correct / total
    
    print(f"\nResults:")
    print(f"Total test images: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    
    # Plot similarity distribution
    plt.figure(figsize=(10, 6))
    plt.hist(similarities, bins=30, edgecolor='black')
    plt.title('Distribution of Similarity Scores')
    plt.xlabel('Similarity Score')
    plt.ylabel('Count')
    plt.savefig('similarity_distribution.png')
    plt.close()
    
    # Create confusion matrix
    unique_labels = sorted(set(true_labels + pred_labels))
    cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, xticklabels=unique_labels, yticklabels=unique_labels, 
                annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels))
    
    # Save results to file
    with open('recognition_results.txt', 'w') as f:
        f.write(f"Palm Print Recognition Test Results\n")
        f.write(f"================================\n\n")
        f.write(f"Total test images: {total}\n")
        f.write(f"Correct predictions: {correct}\n")
        f.write(f"Accuracy: {accuracy:.2%}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(true_labels, pred_labels))

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Run test
    test_recognition_system(
        data_dir="data/aug",
        test_split=0.2  # Use 20% of images for testing
    )
