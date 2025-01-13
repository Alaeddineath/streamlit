import os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the Hugging Face model and tokenizer
@st.cache_resource  # Cache the model and tokenizer so it doesn't reload every time
def load_huggingface_model():
    model_name = "Dzeloq/nllb-msa-dardja-arb-full-data"
    
    # Get the Hugging Face token from environment variables
    hf_token = os.getenv("HUGGINGFACE_TOKEN2")
    if not hf_token:
        raise ValueError("Hugging Face token not found in environment variables.")
    
    # Use GPU if available, otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the tokenizer and model with the token
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=hf_token).to(device)
    
    return tokenizer, model

# Translation function
def translate_text(
    text, 
    tokenizer, 
    model, 
    src_lang='Dardja_Cyrl', 
    tgt_lang='arb_Arab', 
    a=32, 
    b=3, 
    max_input_length=1024, 
    num_beams=4, 
    **kwargs
):
    # Set source and target languages
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    
    # Tokenize the input
    inputs = tokenizer(
        text, 
        return_tensors='pt', 
        padding=True, 
        truncation=True, 
        max_length=max_input_length
    )

    # Generate translated text
    result = model.generate(
        **inputs.to(model.device),
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        max_new_tokens=int(a + b * inputs.input_ids.shape[1]),
        num_beams=num_beams,
        **kwargs
    )

    # Decode and return the result
    return tokenizer.batch_decode(result, skip_special_tokens=True)[0]

# Initialize session states for input/output
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "output_text" not in st.session_state:
    st.session_state.output_text = ""

# Configure the Streamlit app
st.set_page_config(
    page_title="DzEloq Translator",  # Title in the browser tab
    page_icon="https://raw.githubusercontent.com/Alaeddineath/streamlit/main/DZELOQ_LOGO.png",  # Use the logo for the tab icon
    layout="centered"  # Center everything in the app
)

# Header: Add the logo and title
st.markdown(
    """
    <div style="text-align: center;">
        <img src="https://raw.githubusercontent.com/Alaeddineath/streamlit/main/DZELOQ_LOGO.png" alt="Dzeloq Logo" width="200">
        <h1 style="font-size: 2.5em; color: black;">
            <a href="https://www.flaticon.com/free-icons/translation" title="translation icons" style="text-decoration: none; color: black;">
                DzEloq Translator
            </a>
        </h1>
        <p style="font-size: 1.2em; color: #666;">
            Your bridge between <strong>Algerian Darija</strong> and <strong>Arabic</strong>.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load the tokenizer and model
tokenizer, model = load_huggingface_model()

# App layout: Divide into three columns
col1, col2, col3 = st.columns([1, 0.2, 1])

# Add a placeholder switch button
with col2:
    st.button("↔", help="Arabic to Darija translation is coming soon!")

# Translate button logic
if st.button("Translate"):
    if st.session_state.input_text.strip():  # Check if there's text to translate
        with st.spinner("Translating..."):  # Show a spinner while processing
            st.session_state.output_text = translate_text(
                st.session_state.input_text,
                tokenizer,
                model
            )
        st.success("Translation completed!")  # Success message
    else:
        st.error("Please enter text to translate!")  # Error if no input

# Input box for Darija
with col1:
    st.session_state.input_text = st.text_area(
        "Input (Darija)",
        value=st.session_state.input_text,
        height=150,
    )

# Output box for Arabic
with col3:
    st.text_area(
        "Output (Arabic)",
        value=st.session_state.output_text,
        height=150,
        disabled=True,  # Make it read-only
    )

# Footer
st.markdown("---")
st.caption("Made with ❤️ in Algeria. DzEloq © 2024")
