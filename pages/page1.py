import streamlit as st
st.title("🎯 แนวทางการพัฒนาแอปพลิเคชัน")

st.header("📊 1. การเตรียมข้อมูล")
st.write("""
ในการพัฒนาแอปพลิเคชันนี้ เราใช้ข้อมูล 2 ประเภท:
- **ข้อมูลเกม League of Legends**: ข้อมูลเกี่ยวกับผลการแข่งขันเกม League of Legends ในโหมด Ranked
- **ข้อมูลรูปภาพเสื้อ, กางเกง, และถุงเท้า**: ข้อมูลรูปภาพสำหรับการจำแนกประเภท

ขั้นตอนการเตรียมข้อมูล:
1. **ทำความสะอาดข้อมูล**: ลบข้อมูลที่ผิดปกติหรือไม่จำเป็น
2. **แบ่งข้อมูล**: แบ่งข้อมูลเป็นชุดฝึก (Training Set) และชุดทดสอบ (Test Set)
3. **ปรับขนาดข้อมูล**: ปรับขนาดข้อมูลให้เหมาะสมก่อนส่งเข้าโมเดล
""")

st.subheader("💻 โค้ดตัวอย่าง: การเตรียมข้อมูล")
st.code("""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# แบ่งข้อมูลเป็น Features (X) และ Target (y)
X = data.drop(columns=[target_column])
y = data[target_column]

# แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ปรับขนาดข้อมูล (Normalization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
""", language="python")

st.header("🧠 2. ทฤษฎีของอัลกอริทึมที่พัฒนา")

st.subheader("🌳 2.1 Random Forest")
st.write("""
**Random Forest** เป็นอัลกอริทึม Ensemble Learning ที่ใช้ต้นไม้ตัดสินใจ (Decision Trees) หลายๆ ต้นมาทำงานร่วมกัน
- **ข้อดี**: ทนต่อ Overfitting, ทำงานได้ดีกับข้อมูลที่มีมิติสูง
- **การใช้งาน**: ใช้ในหน้าที่ 2 สำหรับการวิเคราะห์ข้อมูลเกม League of Legends
""")

st.subheader("💻 โค้ดตัวอย่าง: Random Forest")
st.code("""
from sklearn.ensemble import RandomForestClassifier

# สร้างโมเดล Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)

# ฝึกโมเดล
model.fit(X_train, y_train)

# ทำนายผล
y_pred = model.predict(X_test)
""", language="python")

st.subheader("🧠 2.2 Neural Network")
st.write("""
**Neural Network** เป็นอัลกอริทึมที่เลียนแบบการทำงานของสมองมนุษย์ โดยใช้เลเยอร์ของโหนด (Nodes) ในการเรียนรู้ข้อมูล
- **ข้อดี**: สามารถเรียนรู้รูปแบบที่ซับซ้อนได้ดี
- **การใช้งาน**: ใช้ในหน้าที่ 3 สำหรับการจำแนกรูปภาพเสื้อ, กางเกง, และถุงเท้า
""")

st.subheader("💻 โค้ดตัวอย่าง: Neural Network")
st.code("""
from sklearn.neural_network import MLPClassifier

# สร้างโมเดล Neural Network
model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)

# ฝึกโมเดล
model.fit(X_train, y_train)

# ทำนายผล
y_pred = model.predict(X_test)
""", language="python")

st.header("🚀 3. ขั้นตอนการพัฒนาโมเดลทั้ง 2 ประเภท")

st.subheader("🌳 3.1 Random Forest")
st.write("""
ขั้นตอนการพัฒนาโมเดล Random Forest:
1. **เตรียมข้อมูล**: แบ่งข้อมูลเป็น Features (X) และ Target (y)
2. **สร้างโมเดล**: ใช้ `RandomForestClassifier` จาก Scikit-learn
3. **ฝึกโมเดล**: ฝึกโมเดลด้วยชุดฝึก (`X_train`, `y_train`)
4. **ประเมินโมเดล**: ใช้ชุดทดสอบ (`X_test`, `y_test`) เพื่อวัดความแม่นยำ
""")

st.subheader("🧠 3.2 Neural Network")
st.write("""
ขั้นตอนการพัฒนาโมเดล Neural Network:
1. **เตรียมข้อมูล**: ปรับขนาดรูปภาพเป็น 28x28 พิกเซลและแปลงเป็น grayscale
2. **สร้างโมเดล**: ใช้ `MLPClassifier` จาก Scikit-learn
3. **ฝึกโมเดล**: ฝึกโมเดลด้วยชุดฝึก (`X_train`, `y_train`)
4. **ประเมินโมเดล**: ใช้ชุดทดสอบ (`X_test`, `y_test`) เพื่อวัดความแม่นยำ
""")

st.write("### 📂 เลือกหน้าที่ต้องการ:")
col1, col2 = st.columns(2)
with col1:
    st.page_link("pages/page2.py", label="ไปที่หน้า 2: วิเคราะห์เกม League of Legends", icon="🌍")
with col2:
    st.page_link("pages/page3.py", label="ไปที่หน้า 3: จำแนกรูปภาพเสื้อ, กางเกง, และถุงเท้า", icon="👕")