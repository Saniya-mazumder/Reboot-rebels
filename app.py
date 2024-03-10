from flask import Flask, render_template, request
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

def extract_foreground(image_data):
    # Read the image from file data
    image = cv2.imdecode(np.frombuffer(image_data.read(), np.uint8), cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresholded = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    foreground_mask = np.zeros_like(thresholded)
    cv2.drawContours(foreground_mask, contours, -1, (255), thickness=cv2.FILLED)

    foreground = cv2.bitwise_and(image, image, mask=foreground_mask)

    return foreground

def create_bar(height, width, color):
    bar = np.zeros((height, width, 3), np.uint8)
    bar[:] = color
    red, green, blue = int(color[2]), int(color[1]), int(color[0])
    return bar, (red, green, blue)

def extract_primary_colour(foreground_image):

    height, width, _ = np.shape(foreground_image)
    # print(height, width)

    data = np.reshape(foreground_image, (height * width, 3))
    data = np.float32(data)

    number_clusters = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(data, number_clusters, None, criteria, 10, flags)
    # print(centers)

    font = cv2.FONT_HERSHEY_SIMPLEX
    bars = []
    rgb_values = []

    for index, row in enumerate(centers):
        bar, rgb = create_bar(200, 200, row)
        bars.append(bar)
        rgb_values.append(rgb)

    img_bar = np.hstack(bars)

    return rgb_values[0]


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    image1 = request.files['image1']
    image2 = request.files['image2']

    # Use image1 and image2 in your ML code
    foreground_image1 = extract_foreground(image1)
    primary_colour1 = list(extract_primary_colour(foreground_image1))

    foreground_image2 = extract_foreground(image2)
    primary_colour2 = list(extract_primary_colour(foreground_image2))

    # Load your dataset
    df = pd.read_csv('C:/Users/H P/OneDrive/Desktop/GDSC/FinalData1.csv')

    X = df[['R1', 'G1', 'B1', 'R2', 'G2', 'B2']]
    y = df['Rating']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Combine the primary colors into a list
    rgb_combined = primary_colour1 + primary_colour2

    # Reshape the list into a 2D array
    new_rgb_values = np.array(rgb_combined).reshape(1, -1)

    # Make predictions using the trained model
    predicted_rating = model.predict(new_rgb_values)

    return render_template('index.html', result=f'Predicted Rating: {predicted_rating[0]}')

if __name__ == '__main__':
    app.run(debug=True)
