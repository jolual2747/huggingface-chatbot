from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import DocsJSONLLoader, get_openai_api_key, get_file_path
from typing import List, Any
from langchain.schema import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.chains import LLMChain

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


def run_query(query: str, retriever: VectorStoreRetriever, llm: Any, message_history: Any):

    llm_summarizer = OpenAI(
        model_name = "gpt-3.5-turbo-instruct",
        temperature = 0.2,
        max_tokens = 256
    )
    template = """You are a helpful assistant created to summarize conversations. Following there is a conversation, you have to summarize it in Spanish.
    Conversation: {messages}

    Summary:
    """
    prompt_template = PromptTemplate(template=template, input_variables=["messages"])
    summarizer_chain = LLMChain(llm=llm_summarizer, prompt=prompt_template)
    if len(message_history.messages) > 0: 
        buffer_memory = ConversationBufferMemory(chat_memory=message_history)
        conversation_summary = summarizer_chain.predict(messages = buffer_memory.buffer)
    else:
        conversation_summary = ''
    
    conversation = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = retriever,
        verbose=True
    )

    results = conversation({"question": query, "chat_history":[("summary", conversation_summary)]})
    message_history.add_user_message(message = query)
    message_history.add_ai_message(message=results["answer"])
    print(conversation_summary)
    print(message_history)
    return results["answer"]

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

    message_history = ChatMessageHistory()
    response = run_query(query = input("Ingresa una pregunta: "), retriever=retriever, llm= llm, message_history=message_history)
    print(response)

if __name__ == '__main__':
    main()