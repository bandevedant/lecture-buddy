import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from transformers import AutoTokenizer, AutoModel
import torch

def get_text_from_pdfs(pdfs_data):
    text = ""
    for pdf in pdfs_data:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunks_from_pdfs_text(raw_pdfs_text):
    chunks = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    ).split_text(raw_pdfs_text)
    return chunks

def get_vector_db(text_chunks):
    # Use HuggingFaceInstructEmbeddings for creating embeddings
    model_name = "hkunlp/instructor-large"
    embeddings = HuggingFaceInstructEmbeddings(model_name=model_name)
    # Convert text_chunks to embeddings
    embedded_texts = [embeddings.embed_documents([chunk])[0] for chunk in text_chunks]
    vector_db = FAISS.from_texts(texts=text_chunks, embeddings=embedded_texts)
    return vector_db

def get_follow_up_conversation(vector_db_store):
    # Use an open-source model from Hugging Face for the LLM
    from langchain.llms import HuggingFaceLLM
    llm = HuggingFaceLLM(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})
    
    memory = ConversationBufferMemory(memory_key='chat-history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Hello, I am your lecture buddy", page_icon=":books:")

    # Define the HTML templates for user and bot messages
    global user_template, bot_template
    user_template = """
    <div style="background-color: #DCF8C6; padding: 10px; border-radius: 10px; margin: 5px;">
        <p>{{MSG}}</p>
    </div>
    """
    bot_template = """
    <div style="background-color: #F1F0F0; padding: 10px; border-radius: 10px; margin: 5px;">
        <p>{{MSG}}</p>
    </div>
    """

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple lecture notes")
    user_question = st.text_input("Ask me about your lecture notes")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Add your notes here")
        pdfs_data = st.file_uploader("Give me the notes here", accept_multiple_files=True)

        if st.button("Think over it"):
            with st.spinner("Processing"):
                # 1. Get the PDF text
                raw_pdfs_text = get_text_from_pdfs(pdfs_data)
                # st.write(raw_pdfs_text) was just testing
                # 2. Split text into chunks
                text_chunks = get_chunks_from_pdfs_text(raw_pdfs_text)
                # st.write(text_chunks)
                # 3. Create vector DB and store it in
                vector_db_store = get_vector_db(text_chunks)
                # print(vector_db_store)
                # 4. Follow-up conversation
                st.session_state.conversation = get_follow_up_conversation(vector_db_store)

if __name__ == '__main__':
    main()
