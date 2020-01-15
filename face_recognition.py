# Import libraries
import cv2
import imutils
from imutils.video import WebcamVideoStream, FPS

# Open the trained face recognizer model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('model/model.yml')

# Open the cascade to detect faces
cascade_path = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)

# Capture video using webcam
capture = WebcamVideoStream(src = 0).start()

# Check the FPS of the video source
fps = FPS().start()

# Initialize the font variable
font = cv2.FONT_HERSHEY_PLAIN

# Initialize the id counter
face_id = 0

# Initialize names variables
names = ['None', 'Edwin', 'Febian']

# Define minimum windo size to be recognized as a face
min_width = 20
min_height = 20

while True:
	# Read the frame from the video source
	frame = capture.read()

	# Resize the frame
	frame = imutils.resize(frame, width = 600, height = 400)

	# Flip the frame horizontally
	frame = cv2.flip(frame, 1)

	# Convert each frame to a grayscale image
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect faces from the grayscale frame
	faces = face_cascade.detectMultiScale(
		gray,
		scaleFactor = 1.1,
		minNeighbors = 5,
		minSize = (min_width, min_height)
		)

	# Drawing a rectangle around the face
	max_rectangle = 1
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 178, 46), max_rectangle)

		# Recognize the face
		face_id, loss = recognizer.predict(gray[y:y+h, x:x+w])

		confidence = round(100 - loss)

		# If the confidence is less than 10, the person is unidentified
		if confidence > 0:
			name = names[face_id]
			percent_match = '{0}%'.format(confidence)
		else:
			name = 'Unidentified'
			percent_match = '{0}%'.format(confidence)

		cv2.putText(frame, str(percent_match) + ' - ' + str(name), (x, y - 10), font, 1, ((255, 178, 46)), max_rectangle)

	# Displaying the frame
	cv2.imshow('Face Recognition', frame)

	# Exit on pressing esc
	key = cv2.waitKey(1)
	if key == 27:
		break

	# Update the FPS counter
	fps.update()

# Clean the FPS counter
fps.stop()

# Display the FPS information
time_elapsed = fps.elapsed()
fps_rate = fps.fps()
print('[INFO] Elapsed time: {:.2f} seconds'.format(time_elapsed))
print('[INFO] Approx. FPS: {:.2f} frames per second'.format(fps_rate))

# Clean the program
print('[INFO] Exiting program')
capture.stop()
cv2.destroyAllWindows()
