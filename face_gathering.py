# Import library
import os
import cv2
import imutils
import dlib
from imutils.video import WebcamVideoStream, FPS
from imutils.face_utils import FaceAligner, rect_to_bb

# Capture the video from the webcam
capture = WebcamVideoStream(src = 0).start()

# Initialize face detector
detector = dlib.get_frontal_face_detector()

# Initialize the face predictor function
face_predictor_path = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(face_predictor_path)
face_aligner = FaceAligner(predictor, desiredFaceWidth = 256)

# Create FPS variable
fps = FPS().start()

# Create a face id for each person
face_id = input('Enter a numeric user id: ')
print('[INFO] Creating a Face ID. Please look at the camera.')

# Create a sample counter
sample_size = 25
count = 0

while True:
	# Read the frame from the video source
	frame = capture.read()

	# Resize the frame
	frame = imutils.resize(frame, width = 600, height = 400)

	# Flip the frame horizontally
	frame = cv2.flip(frame, 1)

	# Convert the frame to grayscale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect faces in the grayscale frame
	max_face = 1
	faces = detector(gray, max_face)

	# Save the face image
	max_rectangle = 1
	for face in faces:
		(x, y, w, h) = rect_to_bb(face)

		# Draw a rectangle around the face
		#	cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 178, 46), max_rectangle)
		count += 1

		# Align the grayscale face image
		aligned_face = face_aligner.align(frame, gray, face)

		# Save the image into the dataset folder
		cv2.imwrite('dataset/user_' + str(face_id) + '_' + str(count) + '.png', aligned_face)

	# Display the frame
	cv2.imshow('Face Recognition', frame)

	# Exit on pressing esc or finish gathering samples
	key = cv2.waitKey(1)
	if key == 27:
		break
	elif count >= sample_size:
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

# Clean up the program
print('[INFO] Exiting program')
capture.stop()
cv2.destroyAllWindows()