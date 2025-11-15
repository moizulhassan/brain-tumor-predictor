import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import os
import time

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Brain Tumor Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ Load Model ------------------
model = load_model('model/brain_tumor_balanced_model.keras')
CLASS_LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']  # Define class labels

# ------------------ Sidebar & Theme ------------------
st.sidebar.title("Settings")
theme = st.sidebar.selectbox("Select Theme", ["Light", "Dark"])
color_scheme = st.sidebar.selectbox("Color Scheme", ["Blue-Green", "Red Accent"])

# Dynamic CSS
if theme == "Dark":
    st.markdown(
        """
        <style>
        body {background-color: #121212; color: #fff;}
        .stButton>button {background-color:#28a745; color:white; border-radius:12px; padding:8px 15px; transition:0.3s;}
        .stButton>button:hover {background-color:#007bff; transform:scale(1.05);}
        </style>
        """, unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        body {background-color: #f5f5f5; color: #000;}
        .stButton>button {background-color:#007bff; color:white; border-radius:12px; padding:8px 15px; transition:0.3s;}
        .stButton>button:hover {background-color:#28a745; transform:scale(1.05);}
        </style>
        """, unsafe_allow_html=True
    )

# ------------------ Header ------------------
st.markdown("<h1 style='text-align:center;color:#28a745;'>Brain Tumor Predictor 🧠</h1>", unsafe_allow_html=True)
st.markdown("---")

# ------------------ Image Upload ------------------
uploaded_file = st.file_uploader("Upload MRI Image", type=['jpg','jpeg','png'])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded MRI', use_column_width=True)

    # Preprocess Image to match training size (150x150)
    img_resized = img.resize((150,150))
    img_array = np.array(img_resized)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction Button
    if st.button("Predict"):
        # Show progress bar
        progress_bar = st.progress(0)
        for percent in range(0, 101, 10):
            time.sleep(0.05)  # simulate loading
            progress_bar.progress(percent)

        # Prediction
        pred = model.predict(img_array)
        class_idx = np.argmax(pred, axis=1)[0]
        confidence = np.max(pred)*100
        st.success(f"Prediction: {CLASS_LABELS[class_idx]} ({confidence:.2f}%)")

        # Show probability chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=CLASS_LABELS,
            y=(pred[0]*100),
            marker_color=['#007bff','#28a745','#dc3545','#ffa500'],
            text=[f"{v:.2f}%" for v in pred[0]*100],
            textposition='auto'
        ))
        fig.update_layout(title="Prediction Probabilities", yaxis_title="Confidence (%)",
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          font=dict(color="#28a745" if theme=="Dark" else "#000"))
        st.plotly_chart(fig, use_container_width=True)

# ------------------ Footer ------------------
st.markdown("---")
st.markdown("<p style='text-align:center;font-size:12px;color:gray;'>Developed by Moiz Hassan | Brain Tumor Detection AI</p>", unsafe_allow_html=True)

