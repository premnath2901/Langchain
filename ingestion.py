import os
import ssl
import asyncio

from typing import List, Dict, Any

import certifi
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (Colors, log_error, log_header, log_info, log_success,
                    log_warning)


load_dotenv()

#configure SSL context for fetching web pages
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(
           model="openai/text-embedding-3-small",
           api_key=os.environ["OPENROUTER_API_KEY"],
           base_url="https://openrouter.ai/api/v1",
           show_progress_bar=False,
           chunk_size=50,
           retry_min_seconds=10,
           check_embedding_ctx_length=False,  # send plain text, not OpenAI-specific token arrays
)



# vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
vectorstore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def main():
  """Main async function to ingest data from the web."""
  log_header("DOCUMENTATION INGESTION PIPELINE")


  log_info(
        "🗺️  TavilyCrawl: Starting to crawl the documentation site : https://python.langchain.com/",
        Colors.PURPLE,
    )

      # Crawl the documentation site

  res = tavily_crawl.invoke(
        {
            "url": "https://python.langchain.com/",
            "max_depth": 1,
            "extract_depth": "advanced",
            "instructions": "content on ai agent "
        }
    )
  # TavilyCrawl returns {"base_url", "results": [{"url", "raw_content"}, ...], ...}
  # all_docs = res["results"]
  all_docs = [Document(page_content=page["raw_content"], metadata={"source": page["url"]}) for page in res["results"] if page.get("raw_content")]

  log_success(
        f"🗺️  TavilyCrawl: Crawling completed successfully - {len(all_docs)} pages",
    )






if __name__ == "__main__":
    asyncio.run(main())