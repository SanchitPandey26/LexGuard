from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from vector_database import get_embedding_model
from langchain_core.prompts import ChatPromptTemplate
import re
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# ✅ Step 2: Initialize LLM (Now using Together AI)
llm_model = ChatOpenAI(
    openai_api_key=os.getenv("TOGETHER_API_KEY"),
    openai_api_base="https://api.together.xyz/v1",
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.7,
    max_tokens=512,
)

# ✅ Step 3: Retrieve Docs
def retrieve_docs(query, file_name):
    db_path = f"vectorstore/{file_name}_faiss"
    if not os.path.exists(db_path):
        print(f"Vector DB not found for {file_name}")
        return []
    embedding_model = get_embedding_model()
    vector_store = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
    return vector_store.similarity_search(query)

def get_context(documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    return context

# ✅ Step 4: Answer Query
custom_prompt_template = """
Use the pieces of information provided in the context to answer the user's question.
If you don’t know the answer, just say that you don’t know—do not make up an answer.
Answer only from the context given, if a question is asked and there is no relevant context say that you don't know.
Only provide the direct answer. Do NOT repeat the question or context.
Do NOT add extra explanations or introductory phrases.

Context:
{context}

Question:
{question}

Answer:
"""

def clean_output(output):
    output = re.sub(r'(?i)(?:question:|context:)', '', output).strip()
    if "Answer:" in output:
        output = output.split("Answer:")[-1].strip()
    return output

def answer_query(documents, model, query):
    if not documents:
        return "No relevant documents found for this query."

    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model
    output = chain.invoke({"question": query, "context": context})

    if not output:
        return "No response from the model."

    return clean_output(output.content if hasattr(output, "content") else str(output))
