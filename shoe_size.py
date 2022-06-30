from dis import Instruction
import streamlit as st 
from PIL import Image
from image_processing import e2e
import cv2

if __name__=='__main__':
    st.title("Find your shoe size!")
    instruction = 'Get your shoe size quickly and easily by uploading an image of your foot on a white A4 papersheet.'
    st.write(instruction)

    file = st.file_uploader("Upload your image")

    if file:
        img = Image.open(file)
        st.title("Here is the image you have uploaded")
        img.thumbnail((600,500), Image.ANTIALIAS)
        st.image(img)
        # img_cv = cv2.imread(file)
        # img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        # length, width, size = e2e(img_cv)
        # st.write("Length: {length} cm, width: {width} cm, size: {size}")
