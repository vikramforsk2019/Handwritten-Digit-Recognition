# MINIST_HAND_WRITTEN_DIGIT_PROJECT
# RECOGNIZATION/HANDWRITTEN
How to Develop a Convolutional Neural Network From Scratch for MNIST Handwritten Digit Classification.

# Preview
IT is a dataset of 60,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9.

# Overview
FRONT-END
Technologies used: As it is a web application,the front-end of this project has been created using HTML (for layout of the webpages), CSS(for styling of the pages), BOOTSTRAP ( for styling and responsiveness), AJAX ( for dynamically communicating with the server) & JAVA-SCRIPT (for adding activity to the page).

BACK-END
Technologies used: As the backend is totally written in PYTHON programming language, things were pretty much easier to be done because Python has a lot of libraries available with it. We used FLASK for writing the server side work. (Django could also have been a great option if we had a wider spectrum of requirements. 
#
# FEATURES
Challenges faced and their solutions:
 1.Problem while posting an image to the server:
The image, when posted from the client page to the server, lost quality when sent directly. This impacted the prediction as the prediction is totally dependent on the pixel intensity values. Changes in even (1/10 )th of the pixel intensities were affecting the results.
 Solution: The images were first converted into the base64 encoded strings and then were posted to the server. This gave perfectly same images on the server side. Which made the results un-affected. This was done using using toDataURL() method (Already discussed in frontend).
 2. Images received on the server side were incompatible to the model.
 3.Accuracy of the predictions were very poor despite the model being trained very well
# Other resources:

- https://medium.com/@vsg16492cse2016/project-implementation-experience-digit-recognizer-for-handwritten-input-images-through-cnn-d9e4b60711ec
- https://machinelearningmastery.com
- https://medium.com
- [Website Templates](https://colorlib.com/wp/templates/)
