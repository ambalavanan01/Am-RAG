import streamlit as st
import os
import time
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# --- Page Configuration ---
st.set_page_config(page_title="Document AI", page_icon="📚", layout="wide")

# --- Custom CSS (Light Theme + Times New Roman) ---
st.markdown("""
<style>
    /* Global Font and Light Background */
    html, body, [class*="css"] {
        font-family: 'Times New Roman', Times, serif !important;
        background-color: #FAFAFA !important;
        color: #111111 !important;
    }
    
    /* Header Styling */
    h1, h2, h3 {
        font-family: 'Times New Roman', Times, serif !important;
        color: #2C3E50 !important;
        border-bottom: 1px solid #EAEAEA;
        padding-bottom: 10px;
    }
    
    /* Subtle Chat Bubble Styling */
    .stChatMessage {
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        background-color: #FFFFFF !important;
        border: 1px solid #EAEAEA;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #F1F4F9 !important;
        border-right: 1px solid #DDDDDD;
    }
    
    /* Expander Styling (For Sources) */
    .streamlit-expanderHeader {
        font-family: 'Times New Roman', Times, serif !important;
        font-size: 0.9em !important;
        color: #666666 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Configuration ---
UPLOAD_DIR = "./uploads"
DB_DIR = "./faiss_index"

# Hardcoded OpenRouter API Key
os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# Ensure our local storage folders exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# --- Header Area ---
st.title("📚 Professional Document AI")
st.markdown("*Upload your documents and interact with them using an advanced Retrieval-Augmented Generation agent.*")
st.divider()

# --- Sidebar: File Upload & Processing ---
with st.sidebar:
    st.header("Upload Center")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, PPTX, or TXT files", 
        type=["pdf", "docx", "pptx", "txt"], 
        accept_multiple_files=True
    )

    if st.button("Index Documents", type="primary"):
        if uploaded_files:
            # 4. Processing State UI Polish
            with st.status("Building Knowledge Base...", expanded=True) as status:
                st.write("📥 Saving files locally...")
                all_documents = []
                
                for file in uploaded_files:
                    file_path = os.path.join(UPLOAD_DIR, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                    
                    st.write(f"📄 Parsing: {file.name}")
                    try:
                        if file.name.endswith(".pdf"):
                            loader = PyPDFLoader(file_path)
                        elif file.name.endswith(".docx"):
                            loader = Docx2txtLoader(file_path)
                        elif file.name.endswith(".pptx"):
                            loader = UnstructuredPowerPointLoader(file_path)
                        elif file.name.endswith(".txt"):
                            loader = TextLoader(file_path)
                        
                        docs = loader.load()
                        for doc in docs:
                            doc.metadata["source_file"] = file.name
                        all_documents.extend(docs)
                    except Exception as e:
                        st.error(f"Error loading {file.name}: {e}")

                st.write("✂️ Chunking text into readable segments...")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
                chunks = text_splitter.split_documents(all_documents)

                st.write("🧠 Generating secure local embeddings (FAISS)...")
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                
                vector_db = FAISS.from_documents(chunks, embeddings)
                vector_db.save_local(DB_DIR)
                
                status.update(label="Knowledge Base Built Successfully!", state="complete", expanded=False)
                
            st.toast(f"Successfully processed {len(uploaded_files)} file(s)!", icon="✅")
        else:
            st.warning("Please select files to upload first.")

# --- Main Area: Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chats with specific avatars
for message in st.session_state.messages:
    avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        
        # Display sources if we saved them in the session state for this message
        if "sources" in message and message["sources"]:
            with st.expander("📝 View Retrieved Sources"):
                for idx, doc in enumerate(message["sources"]):
                    st.markdown(f"**Source {idx+1}:** {doc.metadata.get('source_file', 'Unknown')}")
                    st.caption(f"_{doc.page_content[:200]}..._")
                    st.divider()

if prompt := st.chat_input("Ask a question about your documents..."):
    # Save and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # Agent Processing
    with st.chat_message("assistant", avatar="🤖"):
        try:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_db = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
            retriever = vector_db.as_retriever(search_kwargs={"k": 4}) 
            
            llm = ChatOpenAI(
                model_name="meta-llama/llama-3.3-70b-instruct", 
                temperature=0.3
            )
            
            system_prompt = (
                "You are an intelligent, professional assistant. Use the following pieces of retrieved "
                "context to answer the user's question accurately. If the answer is not in "
                "the context, politely say 'I cannot find the exact answer based on the uploaded documents.'\n\n"
                "Context:\n{context}"
            )
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])

            with st.spinner("Analyzing documents..."):
                # Retrieve the documents
                docs = retriever.invoke(prompt)
                context_str = "\n\n".join([doc.page_content for doc in docs])
                
                # Format into final prompt
                formatted_prompt = prompt_template.format_messages(context=context_str, input=prompt)
                
                # Get Answer
                response = llm.invoke(formatted_prompt)
                answer = response.content
                
                # Display Answer
                st.markdown(answer)
                
                # Display Sources in Expander
                if docs:
                    with st.expander("📝 View Retrieved Sources"):
                        for idx, doc in enumerate(docs):
                            st.markdown(f"**Source {idx+1}:** {doc.metadata.get('source_file', 'Unknown')}")
                            st.caption(f"_{doc.page_content[:200]}..._")
                            st.divider()
                
                # Save to history with sources attached
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": docs
                })
                
        except Exception as e:
             st.error("Error accessing the database. Please ensure you have indexed documents first.")
             st.write(e)

