import cv2

# Load the pre-trained full body detection classifier
body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Load the video file
video_capture = cv2.VideoCapture('ppbody.mp4')

while True:
    # Read each frame of the video
    ret, frame = video_capture.read()

    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect bodies in the frame
    bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Count the number of detected bodies
    num_people = len(bodies)

    # Draw rectangles around the detected bodies
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the frame with body detections and the number of people
    cv2.putText(frame, f'People Count: {num_people}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Body Detection', frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
video_capture.release()
cv2.destroyAllWindows()
