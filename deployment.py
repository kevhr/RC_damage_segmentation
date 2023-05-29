# make a script for the deployment from a trained model with streamlit
import streamlit as st
import numpy as np
import pandas as pd
import time
import streamlit
from PIL import Image
from processing_image import *


st.title('Reinforced concrete damage segmentation application')

st.markdown("***")

st.subheader('Upload the RC damage image')
option = st.radio('', ('Spalling detection', 'Cracking detection'))
st.write('You selected:', option)

if option == 'Spalling detection':
    st.subheader('Upload the Spalling image')
    uploaded_file = st.file_uploader(' ', accept_multiple_files=False)

    if uploaded_file is not None:
        # Perform your Manupilations (In my Case applying Filters)
        # img = load_preprocess_image(uploaded_file)
        img, spall_pixels, square_pixels, rectangle_pixels, \
            spall_diameter, pt_f, mask_total, spall_area, spall_diameter_predicted = final_function(uploaded_file)

        # img = load_preprocess_image(img)
        st.write("Image Uploaded Successfully")
        st.write(img)
        img = load_process_image(img)
        st.image(img)

        # show the results
        st.write('The spall area is: ', int(spall_area), 'mm^2')
        st.write('The spall diameter is: ', int(spall_diameter_predicted), 'mm')

    else:
        st.write("Make sure you image is in TIF/JPG/PNG Format.")

elif option == 'Cracking detection':
    st.subheader('Upload the Cracking image')
    uploaded_file = st.file_uploader(' ', accept_multiple_files=False)
    if uploaded_file is not None:
        # Perform your Manupilations (In my Case applying Filters)
        # img = load_preprocess_image(uploaded_file)
        img, spall_pixels, square_pixels, rectangle_pixels, spall_diameter, pt_f, mask_total, spall_area, spall_diameter_predicted = final_function(uploaded_file)
        # img = load_preprocess_image(img)
        # st.write("Image Uploaded Successfully")
        st.write(img)
        img = load_process_image(str(img))

        st.image(img)

    else:
        st.write("Make sure you image is in TIF/JPG/PNG Format.")

st.markdown("***")

# st.write(' Try again with different inputs')

result = st.button(' Try again')
if result:
    uploaded_file = st.empty()
    predict_button = st.empty()
    streamlit.cache_resource.clear()