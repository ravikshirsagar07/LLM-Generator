import streamlit as st
from transformers import pipeline

# Page Configuration
st.set_page_config(
    page_title="Local LLM Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Local AI Assistant")
st.write("Ask any question. Runs locally without API keys.")

# Load Model
@st.cache_resource
def load_model():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        device_map="auto"
    )

generator = load_model()

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

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

            instruction = f"""
            You are a helpful AI assistant.
            Answer accurately and concisely.

            Question: {prompt}

            Answer:
            """

            result = generator(
                instruction,
                max_length=512,
                temperature=0.3,
                do_sample=True,
                top_p=0.9
            )

            answer = result[0]["generated_text"]

            st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
