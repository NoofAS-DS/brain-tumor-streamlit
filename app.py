# app.py
import io
import numpy as np
import streamlit as st
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from model import CNN_TUMOR

# ===============================
# Config & Constants
# ===============================
IMG_SIZE = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CLASS_NAMES = ["Brain Tumor", "Healthy"]

# ===============================
# Image preprocessing
# ===============================
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ===============================
# Load model (cached)
# ===============================
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = CNN_TUMOR(img_size=IMG_SIZE, num_classes=len(CLASS_NAMES))
    state = torch.load("weights/Brain_Tumor_best_state_dict.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device

# ===============================
# Prediction
# ===============================
def predict(image_pil, model, device):
    x = val_transform(image_pil).unsqueeze(0)
    x = x.to(device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))

    return pred_idx, probs

# ===============================
# Page setup
# ===============================
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor Detection")
st.caption("AI-powered MRI image analysis for educational awareness only")

# ===============================
# Tabs
# ===============================
tab1, tab2, tab3 = st.tabs([
    "📘 About the Disease",
    "🔍 Detection",
    "🩺 Medical Advice"
])

# ===============================
# TAB 1: About
# ===============================
with tab1:
    st.subheader("What is a Brain Tumor?")
    st.write(
        """
        A **brain tumor** is an abnormal growth of cells inside the brain or the central nervous system.
        Tumors can be **benign (non-cancerous)** or **malignant (cancerous)** and may affect brain
        functions depending on their size and location.
        """
    )

    st.subheader("Common Types (General)")
    st.markdown(
        """
        - **Benign tumors**: Slow-growing, less aggressive  
        - **Malignant tumors**: Faster-growing, may spread  
        - **Primary tumors**: Originate in the brain  
        - **Secondary tumors**: Spread from other parts of the body
        """
    )

    st.info(
        "This application does **not** diagnose medical conditions. "
        "It is designed for educational and research demonstration only."
    )

# ===============================
# TAB 2: Detection
# ===============================
with tab2:
    st.subheader("Upload an MRI Image")

    model, device = load_model()

    uploaded = st.file_uploader(
        "Supported formats: JPG, PNG",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded is not None:
        bytes_data = uploaded.read()
        image = Image.open(io.BytesIO(bytes_data)).convert("RGB")

        st.image(image, caption="Uploaded MRI Image", use_container_width=True)

        pred_idx, probs = predict(image, model, device)
        pred_label = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])

        st.divider()
        st.subheader("Result")

        if pred_label == "Brain Tumor":
            st.error(f"Prediction: **{pred_label}**")
        else:
            st.success(f"Prediction: **{pred_label}**")

        st.write(f"Confidence: **{confidence:.2%}**")

        with st.expander("See class probabilities"):
            for name, p in zip(CLASS_NAMES, probs):
                st.write(f"- {name}: {p:.2%}")

# ===============================
# TAB 3: Medical Advice
# ===============================
with tab3:
    st.subheader("When should you see a doctor?")
    st.markdown(
        """
        You should consult a medical professional if you experience:
        - Persistent or severe headaches  
        - Seizures  
        - Vision or speech difficulties  
        - Sudden memory or personality changes  
        - Nausea or vomiting without clear cause  
        """
    )

    st.subheader("Important Notes")
    st.warning(
        """
        - AI predictions are **not a medical diagnosis**  
        - MRI interpretation must be done by qualified radiologists  
        - Always rely on professional medical advice
        """
    )

    st.subheader("Healthy Practice")
    st.markdown(
        """
        - Regular medical check-ups  
        - Early screening when symptoms appear  
        - Following physician recommendations  
        """
    )

st.caption("© Educational AI Demo — Not for clinical use")
