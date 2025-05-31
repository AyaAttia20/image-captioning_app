from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch
import streamlit as st

@st.cache_resource
def load_blip2():
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return processor, model

def generate_blip2_caption(image, processor, model):
    prompt = "Describe this image in detail."
    inputs = processor(images=image, text=prompt, return_tensors="pt")

    # Generate caption with model (device_map="auto" moves model to GPU if available)
    generated_ids = model.generate(**inputs, max_new_tokens=50)
    
    # Decode using the tokenizer inside processor
    caption = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return caption

def main():
    st.title("🖼️ Image Captioning APP")
    st.sidebar.title("About This App 🤗")
    st.sidebar.markdown("""
    This app uses **BLIP-2** with **FLAN-T5 XL** to generate detailed, high-quality captions from uploaded images.
    
    Powered by Hugging Face Transformers and PyTorch.
    """)

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        processor, model = load_blip2()
        with st.spinner("Generating descriptive caption..."):
            caption = generate_blip2_caption(image, processor, model)
            st.success("Generated Caption:")
            st.markdown(
                f"<div style='font-size:20px; font-weight:800; color:#4B4B4B; margin-top:10px;'>{caption}</div>",
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()
