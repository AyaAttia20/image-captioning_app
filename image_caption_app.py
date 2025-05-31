from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch
import streamlit as st

@st.cache_resource
def load_model():
    model_name = "nlpconnect/vit-gpt2-image-captioning"
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, feature_extractor, tokenizer

def generate_caption(image, model, feature_extractor, tokenizer):
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

def main():
    st.title("🖼️ Image Caption Generator")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        model, feature_extractor, tokenizer = load_model()

        with st.spinner("Generating caption..."):
            caption = generate_caption(image, model, feature_extractor, tokenizer)
            st.success("Caption:")
            st.write(f"**{caption}**")

if __name__ == "__main__":
    main()
