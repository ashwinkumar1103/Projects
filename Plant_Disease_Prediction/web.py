import streamlit as st
import tensorflow as tf
import numpy as np

#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr) 
    result_index = np.argmax(prediction)
    return result_index

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Plant Disease Prediction"])

#Home Page
if (app_mode == "Home"):
    st.header("PLANT DISEASE PREDICTION SYSTEM")
    image_path = "home_page.jpeg" 
    st.image(image_path,use_container_width=True)
    st.markdown("""
## 🪴Plant Disease Recognition System 🔍  

Welcome to the **Plant Disease Recognition System**—your AI-powered assistant for **early disease detection** in plants! 🚀  
Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest! 

---

### ❓How It Works  
1️⃣ **Upload an Image** – Navigate to the **Disease Recognition** page and upload a clear image of the affected plant.  
2️⃣ **AI Analysis** – Our deep learning model processes the image and detects any possible diseases.  
3️⃣ **Get Results & Recommendations** – View the identified disease along with suggestions for preventive measures.  

---

### ✅ Why Use This System?  
- 🔬 **Accuracy** – Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.  
- ⚡ **Fast & Efficient** – Get results in seconds, enabling quick decision-making.  
- 🎯 **User-Friendly** – Simple interface for a **hassle-free** experience.  
- 🌾 **Agricultural Impact** – Helps farmers and researchers improve **crop health** and productivity.  

---

### 🚀 Get Started  
Click on **Disease Recognition** in the sidebar to upload an image and get instant results!  

🔹 Want to know more? Visit the **About** page to learn about the project, our team, and our vision for a **greener future**. 🌍  
""")
    
#About Page
elif (app_mode == "About"):
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this github repo. This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure. A new directory containing 33 test images is created later for prediction purpose.
    Idea taken from: https://www.youtube.com/playlist?list=PLvz5lCwTgdXDNcXEVwwHsb9DwjNXZGsoy
    #### Content
    1. Train (70295 images)
    2. Validation (17572 images)
    3. Test (33 images)   

    #### Me
    - **Ashwin Kumar Barla** - B.Tech in Computer Science and Engineering, 2022-26, Kalinga Institute of Industrial Technology (Bhubaneswar, Odisha).
""")
    
#Prediction Page
elif(app_mode=="Plant Disease Prediction"):
    st.header("Plant Disease Prediction")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,use_container_width=True)
    #Predict Button
    if(st.button("Predict")):
        st.write("Prediction")
        result_index = model_prediction(test_image)
        #Define Class
        class_name = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))
