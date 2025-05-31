import streamlit as st
from PIL import Image
from transformers import TFVisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import tensorflow as tf

# Load model and preprocessors once
@st.cache(allow_output_mutation=True)
def load_model():
    model_name = "nlpconnect/vit-gpt2-image-captioning"
    model = TFVisionEncoderDecoderModel.from_pretrained(model_name, from_pt=True)
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, feature_extractor, tokenizer

model, feature_extractor, tokenizer = load_model()

st.title("Image Captioning Demo")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def predict_caption(image):
    pixel_values = feature_extractor(images=image, return_tensors="tf").pixel_values
    output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner('Generating caption...'):
        caption = predict_caption(image)
    st.markdown(f"### Caption: {caption}")
