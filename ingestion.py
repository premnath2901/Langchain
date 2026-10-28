import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader




load_dotenv()

if __name__ == "__main__":
    print("Ingesting....")

    loader = TextLoader("mediumblog1.txt", encoding="utf-8")
    document = loader.load()

    print("splitting...")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")


    # embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    # embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")  # 768 dim -- doesn't match the 1536 index

    # OpenRouter has no DeepSeek embedding model; text-embedding-3-small is 1536 dim, matching the index
    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        check_embedding_ctx_length=False,  # send plain text, not OpenAI-specific token arrays
    )

    print("ingesting...")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ["INDEX_NAME"])
    print("ingested...")
