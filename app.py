from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
model = load_model('mnist_model.h5')

def preprocess_image(file):
    img = Image.open(file).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    return img_array.reshape(1, 28, 28)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        img = preprocess_image(file)
        prediction = model.predict(img)
        digit = np.argmax(prediction)
        return f"Predicted Digit: {digit}"
    return '''
        <form method="post" enctype="multipart/form-data">
            <p>Upload MNIST Image:</p>
            <input type="file" name="file">
            <input type="submit">
        </form>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
