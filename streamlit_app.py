
import streamlit as st 
from PIL import Image
from image_processing import e2e
import cv2

if __name__=='__main__':
    st.title("Find your shoe size!")
    instruction = 'Get your shoe size quickly and easily by uploading an image of your foot on a white A4 sheet of paper. For the best result, please take photo from upper view and capture the whole paper sheet.'
    st.write(instruction)

    file = st.file_uploader("Upload your image")

    if file:
        img = Image.open(file)
        st.title("Here is the image you have uploaded")
        img.thumbnail((600,500), Image.ANTIALIAS)
        st.image(img)
        img.save("img.jpg")
        st.write("Please wait...")
        # OpenCv Read
        img_cv = cv2.imread("img.jpg")
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        length, width, size = e2e(img_cv)
        st.title("Your result:")
        st.write(f"Length: {round(length,1)} cm")
        st.write(f"Width: {round(width,1)} cm")
        st.write(f"Size: {size}")
