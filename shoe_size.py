from dis import Instruction
import streamlit as st 
from PIL import Image

if __name__=='__main__':
    st.title("Find your shoe size!")
    instruction = 'Get your shoe size quickly and easily by uploading an image of your foot on a white A4 papersheet.'
    st.write(instruction)

    file = st.file_uploader("Upload your image")

    if file:
        img = Image.open(file)
        st.title("Here is the image you have uploaded")
        img.thumbnail((400,400), Image.ANTIALIAS)
        st.image(img)
