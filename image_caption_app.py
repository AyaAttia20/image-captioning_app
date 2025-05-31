from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import streamlit as st

@st.cache_resource
def load_blip2():
    processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return processor, model

def generate_blip2_caption(image, processor, model):
    prompt = "Describe this image in detail."
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = processor(image, prompt, return_tensors="pt").to(device, torch.float16)
    output = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

def main():
    st.title("🖼️ Image Captioning APP")
    st.sidebar.title("About This App🤗")
    st.sidebar.markdown("""
    This app uses **BLIP-2** Pre-training with the **FLAN-T5** model to generate high-quality image captions from uploaded images.""")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        processor, model = load_blip2()
        with st.spinner("Generating descriptive caption..."):
            caption = generate_blip2_caption(image, processor, model)
            st.success("Generated Caption:")
            st.markdown(f"<div style='font-size:20px; font-weight:800; color:#4B4B4B; margin-top:10px;'>{caption}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
