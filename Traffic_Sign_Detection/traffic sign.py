import streamlit as st
from PIL import Image
from tensorflow.keras.utils import load_img,img_to_array
import numpy as np
from keras.models import load_model

model = load_model("traffic.h5") # important 
labels = {0:'Bicycles crossing',1:'Children crossing',2:'Danger Ahead',3:'Dangerous curve to the left',4:'Dangerous curve to the right',5:'Dont Go Left',6:'Dont Go Left or Right',7:'Dont Go Right',8:'Dont Go straight',9:'Dont Go straight or left',10:'Dont Go straight or Right',11:'Dont overtake from Left',12:'Fences',13:'Give Way',14:'Go Left',15:'Go Left or right',16:'Go left or straight',17:'Go Right',18:'Go right or straight',19:'Go straight',20:'Go straight or right',21:'Heavy Vehicle Accidents',22:'Horn',23:'Keep Left',24:'keep Right',25:'No Car',26:'No entry',27:'No horn',28:'No stopping',29:'No Uturn',30:'Road Divider',31:'Roundabout mandatory',32:'Speed limit (5kmh)',33:'Speed limit (15kmh)',34:'Speed limit (30kmh)',35:'Speed limit (40kmh)',36:'Speed limit (50kmh)',37:'Speed limit (60kmh)',38:'Speed limit (70kmh)',39:'Speed limit (80kmh)',40:'Traffic signals',41:'Train Crossing',42:'Under Construction',43:'Uturn',44:'Watch out for cars',45:'Zebra Crossing',46:'ZigZag Curve'}

def processed_img(img_path):
    img=load_img(img_path,target_size=(224,224,3))
    img=img_to_array(img)
    img=img/255
    img=np.expand_dims(img,[0])
    answer=model.predict(img)
    y_class = answer.argmax(axis=-1)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    return res

def run():
    st.title("TRAFFIC SIGN DETECTION ")
    img_file = st.file_uploader("Choose an image",type=['jpg','jpeg','png'])

    if img_file is not None :
        img  = Image.open(img_file).resize((250,250))
        st.image(img)
        save_image_path = './upload_image/'+img_file.name
        with open(save_image_path,"wb") as f:
            f.write(img_file.getbuffer())

        if img_file is not None :
            result = processed_img(save_image_path) 
            st.success("**Predicted : "+  result+"**")
run()
