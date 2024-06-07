import streamlit as st
import os
import time
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge import Rouge

# userprompt
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# vectorDB
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings

# llms
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager

# pdf loader
from langchain_community.document_loaders import PyPDFLoader

# pdf processing
from langchain.text_splitter import RecursiveCharacterTextSplitter

# retrieval
from langchain.chains import RetrievalQA

# Verificar y crear directorios si no existen
if not os.path.exists('pdfFiles'):
    os.makedirs('pdfFiles')

if not os.path.exists('vectorDB'):
    os.makedirs('vectorDB')

if not os.path.exists('bitacora'):
    os.makedirs('bitacora')

# Configuración inicial
if 'template' not in st.session_state:
    st.session_state.template = """You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.

    Context: {context}
    History: {history}

    User: {question}
    Chatbot:"""

if 'prompt' not in st.session_state:
    st.session_state.prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=st.session_state.template,
    )

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question",
    )

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = Chroma(persist_directory='vectorDB',
                                          embedding_function=OllamaEmbeddings(base_url='http://localhost:11434',
                                                                             model="llama3")
                                          )

if 'llm' not in st.session_state:
    st.session_state.llm = Ollama(base_url="http://localhost:11434",
                                  model="llama3",
                                  verbose=True,
                                  callback_manager=CallbackManager(
                                      [StreamingStdOutCallbackHandler()]),
                                  )

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'log_file' not in st.session_state:
    log_filename = f"bitacora/chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    st.session_state.log_file = log_filename
    with open(log_filename, 'w') as f:
        f.write("Chatbot Session Log\n")
        f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

def log_message(message):
    with open(st.session_state.log_file, 'a') as log_file:
        log_file.write(message + "\n")

def count_ngrams(sequence, n):
    return {tuple(sequence[i:i+n]): sequence[i:i+n].count(tuple(sequence[i:i+n])) for i in range(len(sequence)-n+1)}

def modified_precision(references, hypothesis, n):
    counts = count_ngrams(hypothesis, n)
    if not counts:
        return 0
    max_counts = {}
    for reference in references:
        reference_counts = count_ngrams(reference, n)
        for ngram in counts:
            if ngram in max_counts:
                max_counts[ngram] = max(max_counts[ngram], reference_counts.get(ngram, 0))
            else:
                max_counts[ngram] = reference_counts.get(ngram, 0)
    clipped_counts = {ngram: min(count, max_counts.get(ngram, 0)) for ngram, count in counts.items()}
    total_counts = sum(counts.values())
    if total_counts == 0:
        return 0
    return sum(clipped_counts.values()) / total_counts

def brevity_penalty(reference, hypothesis):
    ref_len = len(reference)
    hyp_len = len(hypothesis)
    if hyp_len > ref_len:
        return 1
    if hyp_len == 0:
        return 0
    return hyp_len / ref_len

def calculate_bleu(reference, candidate):
    reference = reference.split()
    candidate = candidate.split()
    p1 = modified_precision([reference], candidate, 1)
    p2 = modified_precision([reference], candidate, 2)
    p3 = modified_precision([reference], candidate, 3)
    p4 = modified_precision([reference], candidate, 4)
    bp = brevity_penalty(reference, candidate)
    return bp * (p1 * p2 * p3 * p4) ** 0.25

def calculate_rouge(reference, candidate):
    rouge = Rouge()
    scores = rouge.get_scores(candidate, reference)
    return scores

def calculate_cosine_similarity(reference, candidate):
    vectorizer = TfidfVectorizer().fit_transform([reference, candidate])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0, 1]

st.title("Chatbot - IA - Llama3 RAG - TEC")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])

# Función para procesar y agregar PDF a la base de datos
def process_and_add_pdf(file):
    st.text("File uploaded successfully")
    if not os.path.exists('pdfFiles/' + file.name):
        with st.status("Saving file..."):
            bytes_data = file.read()
            with open('pdfFiles/' + file.name, 'wb') as f:
                f.write(bytes_data)

            loader = PyPDFLoader('pdfFiles/' + file.name)
            data = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200,
                length_function=len
            )

            all_splits = text_splitter.split_documents(data)

            st.session_state.vectorstore = Chroma.from_documents(
                documents=all_splits,
                embedding=OllamaEmbeddings(model="llama3")
            )

            st.session_state.vectorstore.persist()

    st.session_state.retriever = st.session_state.vectorstore.as_retriever()

# Verificar si hay PDFs en la carpeta pdfFiles
pdf_files = [f for f in os.listdir('pdfFiles') if f.endswith('.pdf')]

if not pdf_files and not uploaded_file:
    st.write("Please upload a PDF file to start the chatbot")
else:
    if uploaded_file:
        process_and_add_pdf(uploaded_file)

    st.session_state.retriever = st.session_state.vectorstore.as_retriever()

    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=st.session_state.llm,
            chain_type='stuff',
            retriever=st.session_state.retriever,
            verbose=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": st.session_state.prompt,
                "memory": st.session_state.memory,
            }
        )

    if user_input := st.chat_input("You:", key="user_input"):
        user_message = {"role": "user", "message": user_input}
        st.session_state.chat_history.append(user_message)
        with st.chat_message("user"):
            st.markdown(user_input)

        # Log user question
        log_message(f"User: {user_input}")

        with st.chat_message("assistant"):
            with st.spinner("Assistant is typing..."):
                response = st.session_state.qa_chain(user_input)
            message_placeholder = st.empty()
            full_response = ""
            for chunk in response['result'].split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        # Log assistant response
        log_message(f"Assistant: {response['result']}")

        chatbot_message = {"role": "assistant", "message": response['result']}
        st.session_state.chat_history.append(chatbot_message)

        # Calcular métricas
        bleu_score = calculate_bleu(user_input, response['result'])
        rouge_score = calculate_rouge(user_input, response['result'])
        cosine_sim = calculate_cosine_similarity(user_input, response['result'])
        
        st.write(f"BLEU Score: {bleu_score:.2f}")
        st.write(f"ROUGE Score: {rouge_score}")
        st.write(f"Cosine Similarity: {cosine_sim:.2f}")

# Cerrar el archivo de log al finalizar la sesión
def close_log():
    with open(st.session_state.log_file, 'a') as log_file:
        log_file.write(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

close_log()
