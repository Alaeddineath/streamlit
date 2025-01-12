import os
import streamlit as st
import torch
from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration

@st.cache_resource
def load_huggingface_model():
    model_name = "Aicha-zkr/M2M100-Algerian-Dialect-to-MSA"
    
    # Retrieve the Hugging Face token from the environment
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise ValueError("Hugging Face token not found in environment variables.")
    
    # Set device (CPU or CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model using the token
    tokenizer = M2M100Tokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name, use_auth_token=hf_token).to(device)
    return tokenizer, model

def translate_text(tokenizer, model, text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoded_input = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    arabic_lang_id = tokenizer.get_lang_id("ar")  # Modern Standard Arabic
    generated_tokens = model.generate(
        **encoded_input,
        forced_bos_token_id=arabic_lang_id
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# Initialize session state
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "output_text" not in st.session_state:
    st.session_state.output_text = ""

# Streamlit page config
st.set_page_config(
    page_title="DzEloq",
    page_icon="DZELOQ_LOGO.png",  # Use the uploaded logo as the site icon
    layout="centered"
)

# Title Section with Logo
st.markdown(
    """
    <div style="text-align: center;">
        <img src="DZELOQ_LOGO.png" alt="DZeloq Logo" width="120">
        <h1 style="font-size: 2.5em; color: #4CAF50; margin-top: 10px;"></h1>
        <p style="font-size: 1.2em; color: #666;">
            Your bridge between <strong>Algerian Darija</strong> and <strong>Arabic</strong>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load model and tokenizer
tokenizer, model = load_huggingface_model()

# UI Layout
col1, col2, col3 = st.columns([1, 0.2, 1])

# Switch Button Section
with col2:
    st.button("↔", help="Arabic to Darija translation is coming soon!")

# Translate Button
if st.button("Translate"):
    if st.session_state.input_text.strip():
        with st.spinner("Translating..."):
            st.session_state.output_text = translate_text(
                tokenizer, model, st.session_state.input_text
            )
        st.success("Translation completed!")
    else:
        st.error("Please enter text to translate!")

# Input Section
with col1:
    st.session_state.input_text = st.text_area(
        "Input (Darija)",
        value=st.session_state.input_text,
        height=150,
    )

# Output Section
with col3:
    st.text_area(
        "Output (Arabic)",
        value=st.session_state.output_text,
        height=150,
        disabled=True,
    )

# Footer
st.markdown("---")
st.caption("Made with ❤️ in Algeria. DZeloq © 2024")
