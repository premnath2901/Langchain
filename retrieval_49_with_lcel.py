import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
#
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

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

llm = ChatOpenAI(
    model="deepseek/deepseek-v4-flash",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
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

#===============================================
# Implementation 1 : without LCEL(simple function based approach)
#===============================================


def retrieval_chain_without_lcel(query:str):
   """
   Simple retrieval chain without LCEL 
   Manually retrieves documents, formats them, and invokes the LLM

   Limitations:
   -Manual step by step ececution 
   -No build in streaming support
   -No sync support without additional code 
   -Harder to compose with other chains
   -more verbose and error prone
   """
   #Step 1: Retrieve documents
   docs = retriever.invoke(query)
   print("\n--- Retrieved Documents ---")
   for i, doc in enumerate(docs, start=1):
    print(f"\nDocument {i}:")
    print(doc.page_content)
    print("Metadata:", doc.metadata)
    
   #Step 2: Format documents into existing string
   context = format_docs(docs)

   #Step 3: Format the prompt with context and question
   messages = prompt.format_messages(context=context, question=query)

   #Step 4: Invoke the LLM
   response = llm.invoke(messages)

   #Step 5: Return the response
   return response.content


#===============================================
# Implementation 2 : with LCEL(simple function based approach)
#===============================================
def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain with LCEL
    returns a chain that can be invoked with a query(question)

     Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """

    retrieval_chain =( 
        RunnablePassthrough.assign(context=format_docs) | retriever | format_docs |
        prompt |
        llm |
        StrOutputParser()
    )
    return retrieval_chain





if __name__ == "__main__":
    print("retrieving...")

    #Query 
    # mediumblog1.txt is about vector databases / Milvus -- it never mentions Pinecone,
    # so RAG can only answer questions the blog actually covers
    query  = "What is a vector database and how is it different from a vector search library?"
    
    #===============================================================================
    # Option  0 : Raw invovation with RAG
    #===============================================================================

    print("\n"+ "="*70)
    print("IMPLEMENTATION 0 :Raw LLM Invovation (No RAG )")
    print("="*70)
    result_raw= llm.invoke([HumanMessage(content=query)])
    print("\nAnswer")
    print(result_raw.content)

    #===============================================================================
    # Option  1 : Use implementation without LCEL
    #===============================================================================
    print("\n"+ "="*70)
    print("IMPLEMENTATION 1 : Use implementation without LCEL")
    print("="*70)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer")
    print(result_without_lcel)

    # ========================================================================
    # Option 2: Use implementation WITH LCEL (Better Approach)
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL - Better Approach")
    print("=" * 70)
    print("Why LCEL is better:")
    print("- More concise and declarative")
    print("- Built-in streaming: chain.stream()")
    print("- Built-in async: chain.ainvoke()")
    print("- Easy to compose with other chains")
    print("- Better for production use")
    print("=" * 70)
    print("\nAnswer")
    retrieval_chain = create_retrieval_chain_with_lcel()
    result_with_lcel = retrieval_chain.invoke(query)