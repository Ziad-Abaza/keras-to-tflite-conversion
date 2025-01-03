import tensorflow as tf

# Define your custom objects (if any)
def custom_mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

# Load the Keras model from the HDF5 file with custom objects (if any)
model = tf.keras.models.load_model('1st.h5', custom_objects={'mse': custom_mse})

# Compile the model if necessary (ensure the loss and metrics are the same as during training)
model.compile(optimizer='adam', loss='mse', metrics=['mse'])

# Ensure the model has a defined input shape
input_shape = model.input_shape[1:]  # Get the input shape excluding the batch size

# Define a concrete function with input signature
@tf.function(input_signature=[tf.TensorSpec(shape=[None, *input_shape], dtype=tf.float32)])
def model_func(inputs):
    return model(inputs)

# Convert the Keras model to a TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_concrete_functions([model_func.get_concrete_function()])
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model converted to TFLite and saved as 'model.tflite'")