import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load trained model
model = tf.keras.models.load_model(
    "saved_model/pneumonia_model.h5"
)

# Image path
img_path = "sample.jpeg"

# Load image
img = image.load_img(
    img_path,
    target_size=(224,224)
)

# Convert image into array
img_array = image.img_to_array(img)

# Expand dimensions
img_array = np.expand_dims(
    img_array,
    axis=0
)

# Normalize image
img_array = img_array / 255.0

# Predict
prediction = model.predict(img_array)

# Output result
if prediction[0][0] > 0.5:
    print("PNEUMONIA DETECTED")
else:
    print("NORMAL")