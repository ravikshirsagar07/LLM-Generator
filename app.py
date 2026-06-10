import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="LLM Text Generator",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 LLM Text Generator")
st.write(
    "Enter a prompt below and let the language model generate text."
)

# Load model only once
@st.cache_resource
def load_generator():
    return pipeline(
        "text-generation",
        model="gpt2"
    )

generator = load_generator()

prompt = st.text_area(
    "Enter your prompt:",
    value="Artificial Intelligence will transform education by"
)

max_tokens = st.slider(
    "Maximum new tokens",
    min_value=20,
    max_value=200,
    value=80
)

if st.button("Generate"):
    if prompt.strip():
        with st.spinner("Generating..."):
            result = generator(
                prompt,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95
            )

        st.subheader("Generated Text")
        st.write(result[0]["generated_text"])
    else:
        st.warning("Please enter a prompt.")