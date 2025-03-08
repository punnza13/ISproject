import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.title("🌍 หน้าที่ 2: วิเคราะห์เกม League of Legends")
@st.cache_data 
def load_data(file):
    data = pd.read_csv(file)
    return data

def prepare_data(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y

def train_model(X_train, y_train, algorithm):
    if algorithm == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif algorithm == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif algorithm == "Support Vector Machine (SVM)":
        model = SVC(kernel='linear', random_state=42)
    elif algorithm == "K-Nearest Neighbors (KNN)":
        model = KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError("อัลกอริทึมไม่ถูกต้อง")
    
    model.fit(X_train, y_train) 
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)  
    accuracy = accuracy_score(y_test, y_pred) 
    return accuracy

st.write("### 📤 อัปโหลดไฟล์ CSV ของ League of Legends Ranked Games")

uploaded_file = st.file_uploader("อัปโหลดไฟล์ CSV", type=["csv"])
if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.write("📊 ตัวอย่างข้อมูล:")
    st.write(data.head())

    target_column = st.selectbox("🎯 เลือกคอลัมน์ Target", data.columns)

    X, y = prepare_data(data, target_column)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    algorithm = st.selectbox(
        "🧠 เลือกอัลกอริทึม",
        ["Random Forest", "Logistic Regression", "Support Vector Machine (SVM)", "K-Nearest Neighbors (KNN)"]
    )

    model = train_model(X_train, y_train, algorithm)

    accuracy = evaluate_model(model, X_test, y_test)
    st.success(f"✅ ความแม่นยำของโมเดล ({algorithm}): {accuracy:.2f}")

    st.subheader("🔮 ทำนายผล")
    sample_input = {}
    for column in X.columns:
        sample_input[column] = st.number_input(f"กรุณากรอกค่า {column}", value=0.0)
    if st.button("ทำนาย"):
        sample_df = pd.DataFrame([sample_input])
        prediction = model.predict(sample_df)
        st.write(f"🎯 ผลลัพธ์การทำนาย: **{prediction[0]}**")