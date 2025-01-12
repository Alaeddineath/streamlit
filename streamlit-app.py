import os
import streamlit as st
import torch
from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration

# Load the Hugging Face model and tokenizer
@st.cache_resource  # Cache the model and tokenizer to avoid reloading them on every interaction
def load_huggingface_model():
    # Define the model name
    model_name = "Aicha-zkr/M2M100-Algerian-Dialect-to-MSA"
    
    # Get the Hugging Face token from environment variables
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise ValueError("Hugging Face token not found in environment variables.")
    
    # Set device to CUDA if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the tokenizer and model using the provided Hugging Face token
    tokenizer = M2M100Tokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name, use_auth_token=hf_token).to(device)
    
    return tokenizer, model

# Function to handle translation
def translate_text(tokenizer, model, text):
    
    # Tokenize the input text
    encoded_input = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    
    # Force translation to Modern Standard Arabic (BOS token for Arabic)
    arabic_lang_id = tokenizer.get_lang_id("ar")
    
    # Generate the translated text
    generated_tokens = model.generate(
        **encoded_input,
        forced_bos_token_id=arabic_lang_id
    )
    
    # Decode the tokens to readable text
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# Initialize session state variables (to store user input and output)
if "input_text" not in st.session_state:
    st.session_state.input_text = ""  # Stores the text entered by the user
if "output_text" not in st.session_state:
    st.session_state.output_text = ""  # Stores the translated text

# Configure the Streamlit app's appearance
st.set_page_config(
    page_title="DzEloq Translator",  # Title shown in the browser tab
    page_icon="DZELOQ_LOGO.png",    # Custom logo shown in the browser tab
    layout="centered"               # Center the app layout
)
# Display the DzEloq logo, title, and subtitle with proper alignment and styling
st.markdown(
    """
    <div style="text-align: center;">
        <img src="DZELOQ_LOGO.png" alt="Dzeloq Logo" width="150">
        <h1 style="font-size: 2.5em; color: black; margin-top: 10px;">
            <a href="https://www.flaticon.com/free-icons/translation" title="translation icons">Translation icons created by Dragon Icons - Flaticon</a>
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


# Load the model and tokenizer (cache it to optimize performance)
tokenizer, model = load_huggingface_model()

# UI Layout
col1, col2, col3 = st.columns([1, 0.2, 1])  # Divide the translation section into three columns

# Add a switch button (currently inactive for Darija to Arabic translation)
with col2:
    st.button("↔", help="Arabic to Darija translation is coming soon!")

# Translate Button: Trigger translation when clicked
if st.button("Translate"):
    # Ensure the input text is not empty
    if st.session_state.input_text.strip():
        # Display a spinner while translating
        with st.spinner("Translating..."):
            # Perform the translation
            st.session_state.output_text = translate_text(
                tokenizer, model, st.session_state.input_text
            )
        # Show success message after translation is completed
        st.success("Translation completed!")
    else:
        # Display an error message if the input text is empty
        st.error("Please enter text to translate!")

# Input Section: Add a text box for user input
with col1:
    st.session_state.input_text = st.text_area(
        "Input (Darija)",              # Label for the input box
        value=st.session_state.input_text,  # Pre-fill with any existing session input
        height=150,                    # Height of the text box
    )

# Output Section: Add a disabled text box to show the translated text
with col3:
    st.text_area(
        "Output (Arabic)",             # Label for the output box
        value=st.session_state.output_text,  # Display the translated output
        height=150,                    # Height of the text box
        disabled=True,                 # Make it read-only
    )

# Footer Section: Add a divider and a footer message
st.markdown("---")  # Horizontal divider
st.caption("Made with ❤️ in Algeria. DzEloq © 2024")
