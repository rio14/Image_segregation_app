# This is an Image Segregation app. 

This app can be used to segregate images of one particular person from a folder containing images of multiple people. 

Image segregation is done by creating a model which can distinguish between person X (Images of person you want) from multiple images.
For the training of the algorithm, few images of person X and few images of random people are used.  

### Following steps are performed on all the images present in the training data 

1) Image is read into python as a NumPy array using PIL library 
2) Image is converted into RGB format (incase the image has an alpha channel or is black and white)
3) MTCNN (Multi-Task Cascaded Convolutional Neural Network ) is used to detect and extract a face from the image
4) From the extracted face, a face embedding is created. Face embedding is the vector representation of the image  (Facenet model is used for creating the image embedding)
5) Face embedding is then normalized using Normalized class of scikit learn 
6) The label associated with the image (target variable) is converted to integers using Labelencoder class in scikit-learn.7) Normalized face embedding and associated target variable is used as in input for training of the model

### For segregating images following steps are performed on the images from the folder containing images of person X along with other images: 

1) A single image is first converted into normalized embedding
2) Trained model is used to predict the label of the image
3) If the predicted class belongs to person X then, that image is stored in the "Output Directory" 
4) The function loops over all the images in the folder and stores all the images having person X in the output folder. 


### How to use the app:
1) Download the repository and run the "run.py" file. 
2) The "files" folder has 3 subfolders Train, Test and Output. Training folder contains 2 subfolders (Target and unknown), one with the images of person X(Ben Affelck in this case) and one with random images.
3) Test folder contains images of ben Affleck and random images. Image seggregation will be performed on this folder. 
4) Output folder will be the destination folder where after running the codes, images of ben affleck will be stored. 
5) All the python helper functions for training and testing can be found in the "route.py" file 


#### Pending work: 
I still have to deploy this app. I tried using Render, Heroku for deployment but couldn't figure out how to fix the errors. Any help for the same is appreciated. 


## Credits for the codes:
### 1) Jason Brownlee (https://github.com/jbrownlee). 
I came across the tuorial on "How to Develop a Face Recognition System Using FaceNet in Keras" by Jason Brownlee and have used most of his steps/codes for training and testing purpose. His tutorial (https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/?unapproved=503276&moderation-hash=d69f7dd82f6471c5b50107f71816d44d#comment-503276) is awesome and very informative. All the explantion of the underlying logic of the code used in this app, can be found at the above link. 

### 2) Corey Schafer (https://github.com/CoreyMSchafer) 
I have relied heavily on his series on how to develop a flask application from scratch to learn and understand the basic concepts. I have used his code for adding the Login and Registration functionalities to the app. I would highly recommend his tuorial for flask app development for beginers (https://www.youtube.com/playlist?list=PL-osiE80TeTs4UjLw5MM6OjgkjFeUxCYH). It has helped me a lot to develop this app. 
