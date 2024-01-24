from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import DocsJSONLLoader, get_openai_api_key, get_file_path
from typing import List, Any
from langchain.schema import Document
from langchain_core.vectorstores import VectorStoreRetriever



def load_documents(file_path: str):
    loader = DocsJSONLLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1600,
        length_function = len,
        chunk_overlap = 160
    )

    return text_splitter.split_documents(documents)

def get_chroma_db(embeddings: OpenAIEmbeddings, documents: List[Document], path: str, use_persist_directory=True):
    if use_persist_directory:
        return Chroma(
            persist_directory=path,
            embedding_function=embeddings
        )
    else:
        return Chroma.from_documents(
            documents = documents,
            embedding = embeddings,
            persist_directory = path
        )


def run_query(query: str, retriever: VectorStoreRetriever, llm: Any):
    conversation = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type="stuff",
        retriever = retriever
    )
    return conversation.run(query)

def main():
    documents = load_documents(get_file_path())
    embeddings = OpenAIEmbeddings(model = "text-embedding-ada-002")
    print("Cargando embeddings")
    vector_store = get_chroma_db(
        embeddings,
        documents,
        "chroma_docs"
    )
    retriever = vector_store.as_retriever(
        search_kwargs = {"k":3}
    )
    llm = ChatOpenAI(
            model_name = "gpt-3.5-turbo",
            temperature = 0.2,
            max_tokens = 1000
    )

    response = run_query(query = input("Ingresa una pregunta: "), retriever=retriever, llm= llm)
    print(response)