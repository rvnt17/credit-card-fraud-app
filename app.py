import streamlit as st
import pandas as pd
import joblib
import lightgbm as lgb
from geopy.distance import geodesic
import sqlite3

# --- Styling ---
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #2E3B4E, #1E2A38);
    color: white;
    padding: 20px;
}
.sidebar-title {
    font-size: 24px;
    font-weight: bold;
    color: #F4A261;
    text-align: center;
}
.page-container {
    background: linear-gradient(135deg, #233142, #1E2A38);
    color: white;
    padding: 30px;
    border-radius: 12px;
    box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.3);
    margin: 20px;
}
.page-title {
    font-size: 26px;
    font-weight: bold;
    color: #F4A261;
    text-align: center;
    margin-bottom: 15px;
}
.profile-container {
    background: linear-gradient(135deg, #2E3B4E, #1E2A38);
    color: white;
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.3);
}
</style>
""", unsafe_allow_html=True)

# --- Database Functions ---
def create_users_table():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def add_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
    conn.commit()
    conn.close()

def check_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    result = c.fetchone()
    conn.close()
    return result

# Initialize user database
create_users_table()

# --- Session State for Login ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# --- Login Page ---
def login():
    st.title("🔐 Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if check_user(username, password):
            st.session_state.authenticated = True
            st.success("✅ Logged in successfully!")
            st.rerun()
        else:
            st.error("❌ Invalid credentials.")

    st.markdown("---")
    st.subheader("Register New Account")
    new_user = st.text_input("New Username")
    new_pass = st.text_input("New Password", type="password")
    if st.button("Register"):
        if new_user and new_pass:
            try:
                add_user(new_user, new_pass)
                st.success("🎉 Registered successfully! Please login.")
            except:
                st.warning("⚠️ Username already exists.")
        else:
            st.warning("Please fill in both fields.")

# Show login if not authenticated
if not st.session_state.authenticated:
    login()
    st.stop()

# --- Logout Button ---
with st.sidebar:
    if st.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.rerun()

# --- Load Model and Encoders ---
model = joblib.load("fraud_detection_model.jb")
encoder = joblib.load("label_encoders.jb")

# --- Utility Function ---
def haversine(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

# --- App Pages ---
def home():
    st.title("🏠 Welcome to Fraud Detection System")
    st.subheader("Navigate from the sidebar to:")
    st.markdown("- 🔍 **Check single transactions for fraud**")
    st.markdown("- 📤 **Upload a CSV file for batch prediction**")
    st.markdown("- 👤 **View your profile**")
    st.info("Ensure the model and encoders are loaded to begin.")

def my_profile():
    st.title("👤 My Profile")
    st.markdown('<div class="profile-container">', unsafe_allow_html=True)
    st.subheader("🔹 Name: REVANTH")
    st.subheader("📧 Email: revanth@example.com")
    st.subheader("💼 Role: Fraud Analyst")
    st.subheader("📅 Member since: Jan 2024")
    st.markdown('</div>', unsafe_allow_html=True)

def fraud_check():
    st.title("💳 Check Single Transaction")
    merchant = st.text_input("Merchant Name")
    category = st.text_input("Category")
    amt = st.number_input("Transaction Amount", min_value=0.0, format="%.2f")
    lat = st.number_input("Latitude", format="%.6f")
    long = st.number_input("Longitude", format="%.6f")
    merch_lat = st.number_input("Merchant Latitude", format="%.6f")
    merch_long = st.number_input("Merchant Longitude", format="%.6f")
    hour = st.slider("Transaction Hour", 0, 23, 12)
    day = st.slider("Transaction Day", 1, 31, 15)
    month = st.slider("Transaction Month", 1, 12, 6)
    gender = st.selectbox("Gender", ["Male", "Female"])
    cc_num = st.text_input("Credit Card Number", type="password")

    distance = haversine(lat, long, merch_lat, merch_long)

    if st.button("Check For Fraud"):
        if merchant and category and cc_num:
            input_data = pd.DataFrame([[merchant, category, amt, distance, hour, day, month, gender, cc_num]],
                                      columns=['merchant', 'category', 'amt', 'distance', 'hour', 'day', 'month', 'gender', 'cc_num'])

            for col in ['merchant', 'category', 'gender']:
                try:
                    input_data[col] = encoder[col].transform(input_data[col])
                except:
                    input_data[col] = -1

            input_data['cc_num'] = input_data['cc_num'].apply(lambda x: hash(x) % (10 ** 2))
            prediction = model.predict(input_data)[0]
            result = "Fraudulent Transaction" if prediction == 1 else "Legitimate Transaction"
            st.success(f"Prediction: {result}")
        else:
            st.error("Please fill all required fields.")

def batch_upload():
    st.title("📤 Batch CSV Fraud Detection")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        required_cols = ['merchant', 'category', 'amt', 'lat', 'long', 'merch_lat', 'merch_long', 
                         'hour', 'day', 'month', 'gender', 'cc_num']
        if not all(col in df.columns for col in required_cols):
            st.error(f"CSV must contain columns: {', '.join(required_cols)}")
        else:
            df['distance'] = df.apply(lambda row: haversine(row['lat'], row['long'], row['merch_lat'], row['merch_long']), axis=1)
            df.drop(['lat', 'long', 'merch_lat', 'merch_long'], axis=1, inplace=True)

            for col in ['merchant', 'category', 'gender']:
                try:
                    df[col] = encoder[col].transform(df[col])
                except:
                    df[col] = -1

            df['cc_num'] = df['cc_num'].apply(lambda x: hash(str(x)) % (10 ** 2))
            df['Prediction'] = model.predict(df)
            df['Prediction'] = df['Prediction'].apply(lambda x: "Fraudulent" if x == 1 else "Legitimate")

            st.success("Batch Prediction Complete!")
            st.dataframe(df)
            st.download_button("📥 Download Results", df.to_csv(index=False), "predictions.csv")

# --- Sidebar Navigation ---
st.sidebar.markdown('<p class="sidebar-title">🔎 Navigation</p>', unsafe_allow_html=True)
page = st.sidebar.selectbox("✨ Choose a page:", ["🏠 Home", "🔍 Fraud Check", "📤 Batch Upload", "👤 My Profile"])

if page == "🏠 Home":
    home()
elif page == "🔍 Fraud Check":
    fraud_check()
elif page == "📤 Batch Upload":
    batch_upload()
elif page == "👤 My Profile":
    my_profile()
