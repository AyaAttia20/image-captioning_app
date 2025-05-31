from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import streamlit as st

@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def generate_blip_caption(image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs, max_length=30, num_beams=5)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def main():
    st.title("🖼️ Better Image Caption Generator with BLIP")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        processor, model = load_blip()
        with st.spinner("Generating better caption..."):
            caption = generate_blip_caption(image, processor, model)
            st.success("Caption:")
            st.write(f"**{caption}**")

if __name__ == "__main__":
    main()
