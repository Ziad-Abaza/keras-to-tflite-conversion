# Keras to TensorFlow Lite Conversion

This repository demonstrates how to convert a Keras model into TensorFlow Lite format. It includes the following steps:

1. **Load a Pretrained Keras Model**: The model is loaded from an `.h5` file along with any custom objects (e.g., custom loss functions).
2. **Model Compilation**: Ensures the model is compiled with the appropriate optimizer and loss function.
3. **TensorFlow Lite Conversion**: The model is converted into a TensorFlow Lite format, optimized for mobile and embedded devices.
4. **Saving the Model**: The converted model is saved as a `.tflite` file.

### Requirements
- TensorFlow 2.x or higher

### Usage
Simply run the script to convert your Keras model (`.h5`) into a TensorFlow Lite model (`.tflite`).

```bash
python keras_to_tfLite.py
