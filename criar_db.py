from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
# from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


PASTA_BASE = "base"
PASTA_DESTINO = "destino"

def criar_db():
    documentos = carregar_documentos()
    if not documentos:
        return
    print(documentos)
    chunks = dividir_documentos(documentos)
    vetorizar_chunks(chunks)

def carregar_documentos():
    # carregador = PyPDFDirectoryLoader(PASTA_BASE, glob="*.pdf")  # se fosse .pdf seria por aqui
    carregador = DirectoryLoader(PASTA_BASE, glob="*.txt", loader_cls=TextLoader)
    documentos = carregador.load()
    return documentos

def dividir_documentos(documentos):
    separador_documentos = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=300, 
        length_function=len,
        add_start_index=True
    )
    chunks = separador_documentos.split_documents(documentos)
    print(len(chunks))
    return chunks

# def vetorizar_chunks(chunks):
#     db = Chroma.from_documents(chunks, GoogleGenerativeAIEmbeddings(model="models/embedding-001"), persist_directory="destino")
#     # db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory="destino")
#     print("Banco de dados criado!")

def vetorizar_chunks(chunks):
    # Alteração para que o Gemini gratuito não exceda o limite gratuito
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # 1. Inicializa o banco VAZIO (apenas aponta a pasta)
    db = Chroma(
        persist_directory=PASTA_DESTINO, 
        embedding_function=embedding_model
    )

    # 2. Processa em LOTES para não ser bloqueado
    tamanho_lote = 5  # Envia apenas 5 por vez
    total_chunks = len(chunks)
    
    print(f"Iniciando gravação de {total_chunks} chunks...")
    import time
    for i in range(0, total_chunks, tamanho_lote):
        lote_atual = chunks[i : i + tamanho_lote]
        
        # Adiciona o lote atual
        db.add_documents(lote_atual)
        
        print(f"Processado até chunk {min(i + tamanho_lote, total_chunks)}/{total_chunks}")
        
        # PAUSA OBRIGATÓRIA: Espera 2 segundos para "esfriar" a API
        time.sleep(2)

criar_db()
