import streamlit as st
import time
import os
import textwrap
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain import PromptTemplate

# Page config
st.set_page_config(
    page_title="Gita AI Assistant",
    page_icon="🕉️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stMarkdown {
        text-align: justify;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def initialize_llm():
    """Initialize the Language Model and QA chain"""
    llm = ChatGroq(
        model_name="llama3-8b-8192",
        api_key=st.secrets["GROQ_API_KEY"]
    )
    return llm

def get_embeddings():
    """Initialize embeddings model"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def load_qa_chain():
    """Load the QA chain with FAISS vector store"""
    custom_prompt_template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    
    PROMPT = PromptTemplate(
        template=custom_prompt_template,
        input_variables=['context', 'question']
    )
    
    embeddings = get_embeddings()
    db = FAISS.load_local("db_faiss", embeddings, allow_dangerous_deserialization=True)
    
    return RetrievalQA.from_chain_type(
        llm=initialize_llm(),
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 5}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': PROMPT}
    )

def process_answer(response):
    """Process the LLM response and format it nicely"""
    answer = response['result']
    sources = [os.path.basename(doc.metadata['source']) for doc in response['source_documents']]
    return answer, list(set(sources))

def main():
    # Header
    st.title("🕉️ Bhagavad Gita AI Assistant")
    st.markdown("""
    Welcome to the Bhagavad Gita AI Assistant! Ask any question about the Gita's teachings, 
    philosophy, or specific verses. The assistant uses advanced AI to provide accurate answers 
    based on the sacred text.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("📚 Sample Questions")
        sample_questions = [
            "What is karma yoga according to the Gita?",
            "How does Krishna describe the immortal soul?",
            "What are the main teachings about duty?",
            "Explain the concept of dharma in the Gita",
            "What does the Gita say about meditation?"
        ]
        for q in sample_questions:
            if st.button(q):
                st.session_state.user_input = q

    # Main chat interface
    chat_container = st.container()
    
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"🧑 **You**: {msg['content']}")
            else:
                st.markdown(f"🕉️ **Gita AI**: {msg['content']}")
                if "sources" in msg:
                    st.caption(f"Sources: {', '.join(msg['sources'])}")
    
    # User input
    user_input = st.text_input("Ask your question about the Bhagavad Gita:", 
                              key="user_input", 
                              placeholder="Type your question here...")
    
    if user_input:
        with st.spinner("Consulting the ancient wisdom... 🕉️"):
            try:
                # Record start time
                start_time = time.time()
                
                # Initialize QA chain
                qa_chain = load_qa_chain()
                
                # Get response
                response = qa_chain({"query": user_input})
                answer, sources = process_answer(response)
                
                # Calculate response time
                response_time = round(time.time() - start_time, 1)
                
                # Add to chat history
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                })
                
                # Clear input
                st.session_state.user_input = ""
                
                # Show response time
                st.caption(f"Response time: {response_time}s")
                
                # Rerun to update chat
                st.rerun()
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center'>
        Made with ❤️ | Using Groq LLM and FAISS Vector Store
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
