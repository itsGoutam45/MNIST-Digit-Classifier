import streamlit as st
from PIL import Image, ImageOps
import numpy as np
st.title("MNIST Digit Classifier")

uploaded_file = st.file_uploader(
    "Upload an image of handwritten digit",
    type=["png","jpg","jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.convert("L") #"L" means grayscale pillow
    image_array_check = np.array(image)
#################################
    border_pixels = np.concatenate([
    image_array_check[0, :],
    image_array_check[-1, :],
    image_array_check[:, 0],
    image_array_check[:, -1]
    ])

    if border_pixels.mean() > 127:
        image = ImageOps.invert(image)
    ###########################################
    image = image.resize((28,28))
    st.image(image,width=280)
    
    image_arr = np.array(image)
    st.write("Array shape : ",image_arr.shape)
    st.write("Array data type : ",image_arr.dtype)
    
    image_arr = image_arr.reshape(1,-1)
    st.write("Flattened shape : ",image_arr.shape)
    
    image_arr = image_arr/255.0
    st.write("Minimum pixel value : ",image_arr.min())
    st.write("Maximum pixel value : ",image_arr.max())
    
    #importing pca to use
    import joblib
    pca = joblib.load("mnist_pca.pkl")
    
    image_pca = pca.transform(image_arr)
    st.write("PCA shape : ",image_pca.shape)
    #loading and using svm
    svm = joblib.load("mnist_svm.pkl")
    #making the prediction
    prediction = svm.predict(image_pca)
    st.write("Predict digit : ",prediction[0]) #why predict[0]? 