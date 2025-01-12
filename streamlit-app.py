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
    page_title="DZeloq - Translator",
    page_icon="DZELOQ_LOGO.png",  # Use the uploaded logo as the site icon
    layout="centered"
)

# Custom CSS for Styling
st.markdown(
    """
    <style>
    /* Background color */
    .stApp {
        background-color: #FFFFFF;
    }

    /* Title styling */
    .title-container h1 {
        color: #FF00FF;
        font-size: 2.5em;
        font-weight: bold;
        margin: 10px;
    }

    /* Subtitle styling */
    .subtitle-container p {
        color: #00BFFF;
        font-size: 1.2em;
        margin: 5px;
    }

    /* Input box */
    textarea {
        border: 2px solid #00BFFF;
        border-radius: 5px;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #FF00FF;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        cursor: pointer;
    }
    div.stButton > button:hover {
        background-color: #00BFFF;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #666666;
        font-size: 0.9em;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title Section with Logo
st.markdown(
    """
    <div style="text-align: center;" class="title-container">
        <img src="DZELOQ_LOGO.png" alt="DZeloq Logo" width="120">
        <h1>DZeloq Translator</h1>
        <div class="subtitle-container">
            <p>Seamlessly translate between <strong>Darija</strong> and <strong>Arabic</strong>.</p>
        </div>
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
st.markdown(
    """
    <div class="footer">
        Made with ❤️ in Algeria. DZeloq © 2024
    </div>
    """,
    unsafe_allow_html=True,
)
