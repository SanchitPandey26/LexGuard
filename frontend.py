import streamlit as st
import re
import threading
from rag_pipeline import answer_query, retrieve_docs, llm_model
from vector_database import upload_pdf, process_pdf
import os

# Inject custom CSS for chat container
st.markdown(
    """
    <style>
    .chat-container {
        max-width: 800px;
        margin: auto;
        padding: 10px;
        max-height: 500px;
        overflow-y: auto;
        border: 1px solid #ddd;
        background-color: #f9f9f9;
        direction: rtl;
    }
    .chat-container > div {
        direction: ltr;
    }
    .user-message {
        background-color: #DCF8C6;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        text-align: right;
    }
    .ai-message {
        background-color: #F1F0F0;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar with PDF uploader
with st.sidebar:
    uploaded_file = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=False)
    st.header("Instructions")
    st.markdown(
        """
        - **Upload a PDF:** Upload the Declaration PDF.
        - **Enter your prompt:** Ask legal-related queries.
        - **Chat:** Engage in a legal conversation.
        """
    )


    def run_processing_thread(file_path, db_path):
        def background():
            process_pdf(file_path, db_path)
            with open(os.path.join(db_path, ".completed"), "w") as f:
                f.write("done")  # marker to indicate completion

        thread = threading.Thread(target=background)
        thread.start()
        st.session_state["upload_status"] = "Processing started in the background..."


    if uploaded_file:
        upload_pdf(uploaded_file)
        file_path = "pdfs/" + uploaded_file.name
        db_path = f"vectorstore/{uploaded_file.name}_faiss"

        completed_flag = os.path.join(db_path, ".completed")

        if not os.path.exists(completed_flag):
            if "upload_status" not in st.session_state or "started" not in st.session_state.upload_status:
                run_processing_thread(file_path, db_path)
            st.info(st.session_state.get("upload_status", "Processing..."))
        else:
            st.success(f"{uploaded_file.name} has been successfully processed!")

st.title("LexGaurd")

with st.form(key="chat_form", clear_on_submit=True):
    user_query = st.text_area("Enter your prompt:", height=150, placeholder="Ask Anything!")
    submit_button = st.form_submit_button("Send")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def parse_response(response_text: str):
    """ Extracts only the final answer, ignoring chain-of-thought. """
    final_answer = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
    return final_answer

if submit_button:
    if not uploaded_file:
        st.error("Kindly upload a valid PDF file first!")
    elif not user_query.strip():
        st.error("Please enter your query before sending!")
    else:
        st.chat_message("user").write(user_query)

        # Retrieve documents
        retrieved_docs = retrieve_docs(user_query, uploaded_file.name)

        # Get AI response
        response_text = answer_query(documents=retrieved_docs, model=llm_model, query=user_query)
        final_answer = parse_response(response_text)

        # Save chat history
        st.session_state.chat_history.append({"role": "user", "message": user_query})
        st.session_state.chat_history.append({"role": "LexGuard", "message": final_answer})

st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-message"><strong>You:</strong><br>{msg["message"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="ai-message"><strong>LexGuard:</strong><br>{msg["message"]}</div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
