import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Function to load the translation model
@st.cache_resource
def load_translation_model(direction):
    # Load the appropriate model based on the direction
    model_name = "hanaafra/nllb-msa-dardja-v1"
    
    st.write(f"Loading model: {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Function to translate text
def translate_text(tokenizer, model, text):
    # Tokenize the input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    # Generate translation
    outputs = model.generate(**inputs)
    # Decode the output
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

# Initialize session state for translation direction and text
if "direction" not in st.session_state:
    st.session_state.direction = "Arabic to Darija"
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "output_text" not in st.session_state:
    st.session_state.output_text = ""

# App Layout
st.set_page_config(page_title="DZeloq - Translator", page_icon="🌍", layout="centered")
st.title("🌍 DZeloq")
st.markdown("""
Translate between **Arabic** and **Darija** seamlessly. Use the **Switch Button ↔** to toggle translation direction dynamically.
""")

# Layout for Input, Switch Button, and Output
col1, col2, col3 = st.columns([1, 0.2, 1])

# Switch Button Section
with col2:
    if st.button("↔", help="Switch translation direction"):
        # Toggle the translation direction
        if st.session_state.direction == "Arabic to Darija":
            st.session_state.direction = "Darija to Arabic"
        else:
            st.session_state.direction = "Arabic to Darija"

        # Swap the input and output texts
        temp = st.session_state.input_text
        st.session_state.input_text = st.session_state.output_text
        st.session_state.output_text = temp

# Input Section
with col1:
    input_label = (
        "Input (Arabic)" if st.session_state.direction == "Arabic to Darija" else "Input (Darija)"
    )
    st.session_state.input_text = st.text_area(
        input_label,
        value=st.session_state.input_text,
        height=150,
        key="input_text_area",
    )

# Output Section
with col3:
    output_label = (
        "Output (Darija)" if st.session_state.direction == "Arabic to Darija" else "Output (Arabic)"
    )
    st.text_area(
        output_label,
        value=st.session_state.output_text,
        height=150,
        key="output_text_area",
        disabled=True,
    )

# Translation Button
if st.button("Translate"):
    if st.session_state.input_text.strip():
        with st.spinner("Translating..."):
            # Load the model and tokenizer
            direction = st.session_state.direction
            tokenizer, model = load_translation_model(direction)
            if tokenizer and model:
                # Perform translation and update the output text
                st.session_state.output_text = translate_text(tokenizer, model, st.session_state.input_text)
                st.success("Translation completed!")
            else:
                st.error("Failed to load the translation model.")
    else:
        st.error("Please enter text to translate!")

# Footer
st.markdown("---")
st.caption("Made with ❤️ in Algeria. DZeloq © 2024")
