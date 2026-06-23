import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Page Config
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Assistant")
st.write("Ask any question and get an AI-generated answer.")

# Load Model
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-small"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model

tokenizer, model = load_model()

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Previous Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
prompt = st.chat_input("Ask a question...")

if prompt:

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:
                input_text = f"Answer this question: {prompt}"

                inputs = tokenizer(
                    input_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9
                )

                answer = tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                )

            except Exception as e:
                answer = f"Error: {str(e)}"

            st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
