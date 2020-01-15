# Import library
import os
import cv2
import numpy as np
from PIL import Image

# Path for face image dataset
path = 'dataset'

# Open the face detection cascades
cascade_path = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)

# Initialize the face recognizer function
# Use LBPH(Local Binary Patterns Histograms) Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Function to get the images dataset
def get_images(path):
	image_paths = [os.path.join(path, face) for face in os.listdir(path)]
	face_samples = []
	ids = []

	for image_path in image_paths:
		# Skip the hidden file
		if image_path == path + '/.DS_Store':
			continue

		# Open image using Pillow library
		pillow_image = Image.open(image_path).convert('L') # Convert image to grayscale
		numpy_image = np.array(pillow_image, 'uint8')

		# Fetch the face id from the file name
		id = int(os.path.split(image_path)[-1].split('_')[1])

		# Detect faces
		faces = face_cascade.detectMultiScale(
			numpy_image
			)

		# Appends face image and face id to the lists
		for (x, y, w, h) in faces:
			face_samples.append(numpy_image[y:y+h, x:x+w])
			ids.append(id)

	return face_samples, ids

# Training the face recognizer
print('[INFO] Training faces. Please wait for a while...')
faces, ids = get_images(path)
recognizer.train(faces, np.array(ids))

# Save the model
recognizer.save('model/model.yml')

# Display the number of faces trained
print('[INFO {0} face(s) trained.\nExiting program...'.format(len(np.unique(ids))))

