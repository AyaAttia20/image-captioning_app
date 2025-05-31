from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import streamlit as st

@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.eval()
    return processor, model

def generate_caption(image, processor, model):
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=50)
    caption = processor.tokenizer.decode(out[0], skip_special_tokens=True)
    return caption

def main():
    st.set_page_config(page_title="🖼️ Image Captioning", page_icon="🖼️")
    st.title("🖼️ Image Captioning APP")
    st.sidebar.title("About This App 🤗")
    st.sidebar.markdown("""
    Welcome to the **AI Image Captioning App**
    
    -----------------------🔍🤖---------------------------
    This tool uses artificial intelligence to **automatically describe images** in natural language. 

    -----------------------⚒️📄---------------------------
    Just upload any photo, and the app will instantly generate a short caption explaining what’s in it.""")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        processor, model = load_model()
        with st.spinner("Generating caption..."):
            caption = generate_caption(image, processor, model)
        st.markdown("#### 📸 AI-Generated Description")
        st.markdown(
            f"""
            <div style='
                padding: 1rem;
                background-color: #f0f4f8;
                border-left: 5px solid #0066cc;
                font-size: 17px;
                color: #333;
                font-weight: 600;
                border-radius: 5px;
            '>
                {caption}
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
