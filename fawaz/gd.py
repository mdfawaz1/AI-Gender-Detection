import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import os
import cvlib as cv

# Load the gender detection model
model = load_model('gender_detection.model')

# Open the webcam
webcam = cv2.VideoCapture()

# Classes for gender prediction
classes = ['man', 'woman']

# Directory to save frames with detected gender
save_directory = 'frames_with_gender/'

# Create the directory if it doesn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Loop through frames
while webcam.isOpened():
    # Read frame from the webcam
    status, frame = webcam.read()

    # Apply face detection
    faces, confidences = cv.detect_face(frame)

    # Loop through detected faces
    for idx, face in enumerate(faces):
        (startX, startY, endX, endY) = face

        # Draw bounding box around face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Crop the detected face region
        face_crop = np.copy(frame[startY:endY, startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # Preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = np.expand_dims(face_crop, axis=0)

        # Apply gender detection on face
        conf = model.predict(face_crop)[0]

        # Get label with max accuracy
        gender_idx = np.argmax(conf)
        predicted_gender = classes[gender_idx]

        # Display confidence level
        label = f"{predicted_gender}: {conf[gender_idx]*100:.2f}"
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Save the cropped face if gender is detected
        if conf[gender_idx] > 0.7:  # You can adjust the confidence threshold
            filename = f"face_with_{predicted_gender}_{idx}.jpg"
            filepath = os.path.join(save_directory, filename)

            # Convert the face_crop from [0, 1] range to [0, 255]
            face_crop *= 255.0

            # Ensure values are within the valid range for saving images
            face_crop = np.clip(face_crop, 0, 255).astype(np.uint8)

            # Save the image
            cv2.imwrite(filepath, face_crop[0])

    # Display output
    cv2.imshow("gender detection", frame)

    # Press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()