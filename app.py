import streamlit as st

st.title("🚀 SE Project: แอปพลิเคชันหลายหน้า")
st.write("---")
col1, col2 = st.columns([1, 3])
with col1:
    st.image("https://via.placeholder.com/150", width=150)  # แทนที่ด้วยโลโก้ของโปรเจกต์
with col2:
    st.markdown("""
    <h2 style='color: #FF5733;'>ยินดีต้อนรับสู่ SE Project!</h2>
    <p style='font-size: 18px;'>
        โปรเจกต์นี้พัฒนาขึ้นเพื่อการเรียนรู้และทดลองใช้ Machine Learning กับข้อมูลจริง
        โดยมี 2 หน้าที่คุณสามารถใช้งานได้:
    </p>
    """, unsafe_allow_html=True)

st.write("---")
st.header("📂 หน้าที่มีในแอปพลิเคชัน")

with st.expander("🌍 หน้าที่ 2: วิเคราะห์เกม League of Legends"):
    st.write("""
    - **วัตถุประสงค์**: วิเคราะห์ข้อมูลเกม League of Legends และทำนายผลการแข่งขัน
    - **เทคโนโลยีที่ใช้**: Random Forest จาก Scikit-learn
    - **ขั้นตอนการใช้งาน**:
        1. อัปโหลดไฟล์ CSV ของข้อมูลเกม
        2. เลือกคอลัมน์ Target
        3. ฝึกโมเดลและทำนายผล
    """)

    st.subheader("📊 ที่มาของ Dataset")
    st.write("""
    Dataset นี้มาจากเว็บไซต์ [Kaggle](https://www.kaggle.com/datasets/datasnaek/league-of-legends) 
    ซึ่งเป็นข้อมูลเกี่ยวกับเกม League of Legends ในโหมด Ranked
    """)

    st.subheader("🔍 Feature ของ Dataset")
    st.write("""
    Dataset นี้มี Feature ต่างๆ ที่เกี่ยวข้องกับเกม League of Legends เช่นs:
    - **gameId**: รหัสเกม
    - **creationTime**: เวลาที่สร้างเกม
    - **gameDuration**: ระยะเวลาเกม
    - **seasonId**: รหัสซีซัน
    - **winner**: ทีมที่ชนะ (1 = ทีมสีน้ำเงิน, 2 = ทีมสีแดง)
    - **firstBlood**: ทีมที่ได้ First Blood
    - **firstTower**: ทีมที่ทำลายหอคอยแรก
    - **firstInhibitor**: ทีมที่ทำลาย Inhibitor แรก
    - **firstBaron**: ทีมที่สังหาร Baron แรก
    - **firstDragon**: ทีมที่สังหาร Dragon แรก
    - **firstRiftHerald**: ทีมที่สังหาร Rift Herald แรก
    - **t1_champ1id** ถึง **t2_champ5id**: ตัวละครที่ผู้เล่นเลือกในแต่ละทีม
    """)

with st.expander("👕 หน้าที่ 3: จำแนกรูปภาพเสื้อ, กางเกง, และถุงเท้า"):
    st.write("""
    - **วัตถุประสงค์**: จำแนกรูปภาพเสื้อ, กางเกง, และถุงเท้า
    - **เทคโนโลยีที่ใช้**: Neural Network (MLPClassifier) จาก Scikit-learn
    - **ขั้นตอนการใช้งาน**:
        1. อัปโหลดรูปภาพ
        2. ฝึกโมเดลและทำนายผล
    """)

st.write("---")
st.write("### 📂 เลือกหน้าที่ต้องการ:")
col1, col2 = st.columns(2)
with col1:
    st.page_link("pages/page2.py", label="ไปที่หน้า 2: วิเคราะห์เกม League of Legends", icon="🌍")
with col2:
    st.page_link("pages/page3.py", label="ไปที่หน้า 3: จำแนกรูปภาพเสื้อ, กางเกง, และถุงเท้า", icon="👕")