from keras.models import load_model
import cv2
import numpy as np
from collections import Counter

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model(r"C:\Users\hithu\OneDrive\Desktop\Interview_confidence_level\keras_model.h5", compile=False)

model.summary()

# Load the labels
class_names = open(r"C:\Users\hithu\OneDrive\Desktop\Interview_confidence_level\labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on the default camera of your computer
camera = cv2.VideoCapture(0)
# Increase the window size
cv2.namedWindow("Webcam Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Webcam Image", 800, 600)  # Set the width and height as desired

camera.set(3, 640)  # Set the width
camera.set(4, 480)  # Set the height

# Create a list to store the predicted classes
predicted_classes = []

while True:
    # Grab the web camera's image
    ret, image = camera.read()

    image = cv2.flip(image, 1)

    # Resize the raw image into (224-height, 224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array
    image = np.asarray(image, dtype=np.uint8)

    # Predict the model
    prediction = model.predict(np.expand_dims(image, axis=0) / 255.0)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Append the predicted class to the list
    predicted_classes.append(class_name)

    # Prepare text for display
    class_text = "Class: " + class_name[2:]
    confidence_text = class_text + ":" + str(np.round(confidence_score * 100))[:-2] + "%"

    # Draw the text on the frame
    # cv2.putText(image, class_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(image, confidence_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Create a confidence bar
    confidence_bar_length = int(confidence_score * 150)
    cv2.rectangle(image, (10, 80), (10 + confidence_bar_length, 90), (0, 255, 0), -1)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Listen to the keyboard for key presses
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()

print(predicted_classes)

percentage_confidence = predicted_classes.count('0 Confident\n') / len(predicted_classes) * 100

print(percentage_confidence)

if percentage_confidence < 50:
    print('Confidence Level : Low')
elif 50 <= percentage_confidence < 75:
    print('Confidence Level : Medium')
else:
    print('Confidence Level : High')