import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("initilizing...")

embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        check_embedding_ctx_length=False,  # send plain text, not OpenAI-specific token arrays
    )

vectorstore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


prompt = ChatPromptTemplate.from_template(
    """  Answer the question based only on the following context:
    {context}
    Question: {question}

    provide the detailed answer 

     """)

def format_docs(docs):
  """ Format the documents for the prompt """
  return "\n\n".join([doc.page_content for doc in docs])


   
if __name__ == "__main__":
    print("retrieving...")

    #Query 
    query  = "What is  pinecone in machine learning?"
    
    #===============================================================================
    # Option  0 : Raw invovation with RAG
    #===============================================================================

    print("\n"+ "="*70)
    print("IMPLEMENTATION 0 :Raw LLM Invovation (No RAG )")
    print("="*70)
    result_raw= llm.invoke(HumanMessage(content=query))
    print("\nAnswer")
    print(result_raw.content)