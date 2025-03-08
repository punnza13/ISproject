import streamlit as st
import numpy as np
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("👕 หน้าที่ 3: จำแนกรูปภาพเสื้อ, กางเกง, และถุงเท้า")
class_names = ["เสื้อ", "กางเกง", "ถุงเท้า"]

def preprocess_image(img):
    img = img.resize((28, 28)) 
    img = img.convert("L")  
    img_array = np.array(img).flatten() 
    return img_array

st.write("### 📤 อัปโหลดรูปภาพเสื้อ, กางเกง, หรือถุงเท้า")

uploaded_file = st.file_uploader("อัปโหลดรูปภาพ", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="รูปภาพที่อัปโหลด", use_container_width=True)

    img_array = preprocess_image(img)

    X_train = np.random.rand(100, 784)  
    y_train = np.random.randint(0, 3, 100)  

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    model.fit(X_train, y_train) 

    img_array_scaled = scaler.transform([img_array]) 
    predicted_class = class_names[model.predict(img_array_scaled)[0]]
    st.success(f"🎯 ผลลัพธ์การทำนาย: **{predicted_class}**")