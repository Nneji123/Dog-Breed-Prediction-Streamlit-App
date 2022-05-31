#Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import joblib

#Loading the Model
model = joblib.load('dog_breed_predictor.pkl')


st.markdown("## Dog Breed Prediction App")
st.markdown("""
This app uses deep learning (Convolutional Neural Network) libraries namely keras to predict the following breeds of dogs given the input image:

1. Scottish Deerhound
2. Maltese Dog
3. Afghan Hound
4. Entlebucher
5. Bernese Mountain Dog

**Made by Ifeanyi Nneji**


Data source: kaggle/catherinehorng/dogbreedidfromcomp
""")

#Name of Classes
CLASS_NAMES = ['Scottish Deerhound' ,'Maltese Dog' ,'Afghan Hound ','Entlebucher ','Bernese Mountain Dog']

#Uploading the dog image
dog_image = st.file_uploader("Upload an image of the dog...", type=['png','jpg','webp','jpeg'])
submit = st.button('Predict')
#On predict button click
if submit:

    if dog_image is not None:

        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(dog_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)



        # Displaying the image
        st.image(opencv_image, channels="BGR")
        #Resizing the image
        opencv_image = cv2.resize(opencv_image, (224,224))
        #Convert image to 4 Dimension
        opencv_image.shape = (1,224,224,3)
        #Make Prediction
        Y_pred = model.predict(opencv_image)

        st.text(str("The Dog Breed is "+CLASS_NAMES[np.argmax(Y_pred)]))
