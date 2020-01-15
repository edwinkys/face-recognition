# Face Recognition
This repository contains the codes for creating a facial recognition program using open source libraries such as OpenCV and several other libraries. This facial recognition program will take video input from your default webcam, read each frame, convert it to grayscale, detect faces, and recognize those faces.

# Running
First, you will need to install all of the dependencies, libraries, such as OpenCV, Dlib, Imutils, and other libraries.

After installing all of the dependencies, you can start by running `face_gathering.py` file that contains the code to collect your face data through a camera. Don't forget to change the list of names of the person in the data based on each id.

Then, you have to train the face recognition algorithm with the face data that you have collected. To do this, you need to run `face_training.py` file. This code will save the trained model as `model.yml`.

After training the algorithm, you can run the `face_recognition.py` program. 
