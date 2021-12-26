import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import time
fig = plt.figure()

with open("custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title('Pepper_Bell Health check')

st.markdown("This application simply tells whether leaf is healthy or not")


def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    class_btn = st.button("Predict")
    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                plt.imshow(image)
                plt.axis("off")
                predictions = predict(image)
                time.sleep(1)
                st.success('Predicted')
                st.write(predictions)
                



def read_file_as_image(data) -> np.ndarray:
    image = np.array(data)
    return image


def predict(image):
    classifier_model = "pepper_bell.h5"
    model = load_model(classifier_model, compile=False, custom_objects={'KerasLayer': hub.KerasLayer})
    class_names = [
          'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy']

    image = read_file_as_image(image)
    img_batch = np.expand_dims(image, 0)


    predictions = model.predict(img_batch)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)

    
    result = f"{predicted_class} with a { confidence } % confidence." 
    return result







    

if __name__ == "__main__":
    main()


