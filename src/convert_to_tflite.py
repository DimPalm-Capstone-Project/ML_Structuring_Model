import tensorflow as tf
import os
from predict import PalmPrintRecognizer

def convert_to_tflite():
    # Initialize model
    model_path = 'src/models/palm_print_siamese_model.h5'
    recognizer = PalmPrintRecognizer(model_path)
    model = recognizer.model
    
    # Convert the model to TFLite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Apply optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    # Enable quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    
    # Convert model
    tflite_model = converter.convert()
    
    # Save the model
    tflite_model_path = 'src/models/palm_print_model.tflite'
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Original model size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    print(f"TFLite model size: {os.path.getsize(tflite_model_path) / (1024*1024):.2f} MB")
    
    return tflite_model_path

def test_tflite_model(tflite_model_path):
    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("\nModel Input Shape:", input_details[0]['shape'])
    print("Model Output Shape:", output_details[0]['shape'])
    
    # Test with dummy input
    input_shape = input_details[0]['shape']
    dummy_input = tf.random.uniform(input_shape)
    
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()
    
    output = interpreter.get_tensor(output_details[0]['index'])
    print("\nTest inference successful!")
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    print("Converting model to TFLite format...")
    tflite_path = convert_to_tflite()
    
    print("\nTesting TFLite model...")
    test_tflite_model(tflite_path)
