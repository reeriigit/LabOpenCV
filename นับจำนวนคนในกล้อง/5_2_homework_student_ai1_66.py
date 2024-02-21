import cv2
import os

# Load Haar Cascade XML file
xml_file_path = "haarcascade_frontalface_default.xml"
if not os.path.exists(xml_file_path):
    print(f"Error: The file {xml_file_path} does not exist.")
    exit()

# Open the video capture (assuming the default camera)
cap = cv2.VideoCapture(0)

# Load the face image to be resized
overlay_path = "fuckyou.jpg"
if not os.path.exists(overlay_path):
    print(f"Error: The file {overlay_path} does not exist.")
    exit()   

# Read the face image
overlay_img = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)

# Resize the face image to 400x400
resized_overlay_img = cv2.resize(overlay_img, (400, 400))

# Load Haar Cascade XML file
face_cascade = cv2.CascadeClassifier(xml_file_path)

# Counter for detected faces
face_count = 0

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Perform face detection
    faces = face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 10)

    # If faces are detected
    if len(faces) > 0:
        # Update the face count
        face_count = len(faces)

        # Display the count of detected faces on the image
        cv2.putText(frame, f"Number of Faces: {face_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0,0 ), 5, cv2.LINE_AA)

        # Iterate through all detected faces
        for (x, y, w, h) in faces:
            # Resize the overlay image to the size of the detected face
            resized_overlay_img = cv2.resize(overlay_img, (w, h))

            # Place the resized overlay image on top of the original image
            frame[y:y+h, x:x+w] = cv2.addWeighted(frame[y:y+h, x:x+w], 1, resized_overlay_img[:, :, :3], 0.5, 0)

    # Display the result
    cv2.imshow('Result', frame)

    # Check for the 'q' key to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
