# app.py
import io
import numpy as np
import streamlit as st
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from model import CNN_TUMOR

IMG_SIZE = 256
CLASS_NAMES = ["Brain Tumor", "Healthy"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

@st.cache_resource
def load_model():
    model = CNN_TUMOR(img_size=IMG_SIZE, num_classes=len(CLASS_NAMES))
    state = torch.load("weights/Brain_Tumor_best_state_dict.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

def predict(image, model):
    x = transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).numpy()[0]
    return probs

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

st.title("🧠 Brain Tumor Detection")
st.caption("Educational demo only – not for medical diagnosis.")

model = load_model()

uploaded = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    probs = predict(image, model)
    pred = np.argmax(probs)

    st.subheader("Prediction")
    st.write(f"**{CLASS_NAMES[pred]}**")
    st.write(f"Confidence: **{probs[pred]*100:.2f}%**")

    st.subheader("Class Probabilities")
    for name, p in zip(CLASS_NAMES, probs):
        st.write(f"{name}: {p*100:.2f}%")
