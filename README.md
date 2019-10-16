# This is an Image Segregation app. 

This app can be used to segregate images of one particular person from a folder containing images of multiple people. 

Image segregation is done by creating a model which can distinguish between person X (Images of person you want) from multiple image.
For the training of the algorithm, Few images of person X and few images of random people are used.  

### Following steps are performed on all the images present in the training data 

1) Image is read into python as a NumPy array using PIL library 
2) Image is converted into RGB format (incase the image has an alpha channel or is black and white)3) MTCNN (Multi-Task Cascaded        Convolutional Neural Network ) is used to detect and extract a face from the image4) From the extracted face, a face embedding is created. Face embedding is the vector representation of the image  (Facenet model is used for creating the image embedding)
5) Face embedding is then normalized using Normalized class of scikit learn 
6) The label associated with the image (target variable) is converted to integers using Labelencoder class in scikit-learn.7) Normalized face embedding and associated target variable is used as in input for training of the model

### For segregating images following steps are performed on the images from the folder containing images of person X along with other images: 

1) A single image is first converted into normalized embedding2) Trained model is used to predict the label of the image3) If the predicted class belongs to person X then, that image is stored in the "Output Directory" 
4) The function loops over all the images in the folder and stores all the images having person X in the output folder. 
