import cv2
import sys

cascPath = sys.argv[1]
cascade = cv2.CascadeClassifier(cascPath)

webcamCapt = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = webcamCapt.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    objects = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    # Draw a rectangle around the objects
    for (x, y, w, h) in objects:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Wait for 'q' key to exit the webcam view
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
webcamCapt.release()
cv2.destroyAllWindows()