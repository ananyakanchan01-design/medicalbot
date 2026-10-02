import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
load_dotenv(find_dotenv())

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def get_standalone_query(llm, chat_history, current_query):
    """Rephrase follow-up questions to include conversation context for accurate FAISS retrieval."""
    if not chat_history:
        return current_query
    
    # Check if query is short or a follow-up request (e.g., "explain in simple words", "tell me more", "simplify")
    followup_triggers = ["explain", "simple", "more", "why", "how", "what", "simplify", "easier", "summary", "summarize", "detail"]
    if len(current_query.split()) > 7 and not any(t in current_query.lower() for t in ["simple", "simplify", "easier", "above", "that", "this"]):
        return current_query

    history_str = ""
    for msg in chat_history[-4:]:
        role = "User" if msg['role'] == 'user' else "Assistant"
        history_str += f"{role}: {msg['content'][:250]}\n"
    
    rephrase_prompt = f"""Given the following medical conversation history and a user follow-up question, rephrase the follow-up question into a complete, standalone search query focused on the medical topic being discussed.
    
Conversation History:
{history_str}

User Follow-up Question: {current_query}

Output ONLY the rephrased standalone query (no extra words or quotes):"""

    try:
        res = llm.invoke(rephrase_prompt)
        standalone = res.content.strip().strip('"').strip("'")
        if standalone:
            return standalone
    except Exception:
        pass
    return current_query

def main():
    load_dotenv(find_dotenv(), override=True)
    st.set_page_config(page_title="MediBot - Medical AI Assistant", page_icon="🩺", layout="wide")

    # Sidebar configuration
    st.sidebar.title("🩺 MediBot Settings")
    st.sidebar.markdown("Medical Knowledge Base powered by **The Gale Encyclopedia of Medicine** & LLM RAG.")

    groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()

    selected_model = st.sidebar.selectbox(
        "Select LLM Model",
        [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b"
        ],
        index=0
    )

    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Disclaimer**: MediBot provides information from medical encyclopedia sources for educational purposes only. Always consult a healthcare professional for medical advice.")

    # Main Chat Interface
    st.title("🩺 MediBot - AI Medical Assistant")
    st.caption("Ask questions based on verified medical encyclopedia knowledge base.")

    if not groq_api_key:
        st.warning("⚠️ Groq API Key is not set in `.env`. Please add `GROQ_API_KEY=your_key` to your `.env` file.")
        return

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
            if 'sources' in message and message['sources']:
                with st.expander("📚 Source Reference Documents"):
                    for idx, doc in enumerate(message['sources'], 1):
                        st.markdown(f"**Document {idx} (Page {doc.metadata.get('page', 'N/A')}):**")
                        st.caption(doc.page_content[:300] + "...")

    prompt = st.chat_input("Ask a medical question (e.g., What are the symptoms of Asthma?)...")

    if prompt:
        st.chat_message('user').markdown(prompt)

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer user's question.
        If you dont know the answer, just say that you dont know, dont try to make up an answer. 
        Dont provide anything out of the given context

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """
        
        try: 
            with st.spinner("Searching medical knowledge base and generating answer..."):
                vectorstore = get_vectorstore()
                if vectorstore is None:
                    st.error("Failed to load the vector store")
                    return

                llm_instance = ChatGroq(
                    model_name=selected_model,
                    temperature=0.0,
                    groq_api_key=groq_api_key,
                )

                # Rephrase query if it is a follow-up question
                search_query = get_standalone_query(llm_instance, st.session_state.messages, prompt)

                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm_instance,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                )

                response = qa_chain.invoke({'query': search_query})

                result = response["result"]
                source_documents = response.get("source_documents", [])

                with st.chat_message('assistant'):
                    st.markdown(result)
                    if source_documents:
                        with st.expander("📚 Source Reference Documents"):
                            for idx, doc in enumerate(source_documents, 1):
                                st.markdown(f"**Document {idx} (Page {doc.metadata.get('page', 'N/A')}):**")
                                st.caption(doc.page_content[:300] + "...")

                st.session_state.messages.append({'role': 'user', 'content': prompt})
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': result,
                    'sources': source_documents
                })

        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main()