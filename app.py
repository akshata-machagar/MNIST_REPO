from tensorflow.keras.models import load_model
import numpy as np

# Load the saved model
model = load_model('mnist_model.h5')

# Create a dummy input (example: blank MNIST image)
dummy_input = np.random.rand(1, 28, 28)

# Predict using the model
predictions = model.predict(dummy_input)

# Output prediction
print("Predictions:", predictions)
