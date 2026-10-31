import os
import ssl
import asyncio
import hashlib

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



vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
#vectorstore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)
tavily_extract = TavilyExtract()
# tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)  # follows links to GitHub/TikTok etc.
tavily_map = TavilyMap(
    max_depth=3,
    max_breadth=20,
    limit=1000,                                  # "max_pages" is not a TavilyMap option -- it's "limit"
    allow_external=False,                        # stay on docs.langchain.com (no GitHub/TikTok links)
)
tavily_crawl = TavilyCrawl()

#Batching the URLs
def chunk_urls(urls: List[str], chunk_size: int = 3) -> List[List[str]]:
    """Split URLs into chunks of specified size."""
    chunks = []
    for i in range(0, len(urls), chunk_size):
        chunk = urls[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

async def extract_batch(urls: List[str], batch_num: int) -> List[Dict[str, Any]]:
    """Extract documents from a batch of URLs."""
    try:
        log_info(
            f"🔄 Processing batch {batch_num} with {len(urls)} URLs", Colors.BLUE
            )
        docs = await tavily_extract.ainvoke(input={"urls": urls})
        results = docs.get('results', [])
        log_success(
            f"✅ Batch {batch_num} completed - extracted {len(results)} documents", Colors.GREEN
            )
        return results
    except Exception as e:
        log_error(f"❌ Batch {batch_num} failed: {e}", Colors.RED)
        return []


async def async_extract(urls_batches: List[List[str]]) -> List[Document]:
    log_header("DOCUMENTATION INGESTION PIPELINE")
    log_info("🔄 Extracting documents from the documentation site", Colors.BLUE)
    tasks = [extract_batch(batch,i+1) for i, batch in enumerate(urls_batches)]
    results = await asyncio.gather(*tasks)

#=========================================================
# Filter out the pages that are not in the documentation site
#=========================================================
    all_pages =[]
    failed_pages =0
    # Each item in results is the list of pages extracted_batch() returned for one batch
    for batch_num, batch_results in enumerate(results, start=1):
       if isinstance(batch_results, Exception):
         log_error(f"❌ Batch {batch_num} failed: {batch_results}", Colors.RED)
         failed_pages += 1
       else:
        for extracted_page in batch_results:
            if not extracted_page.get("raw_content"):  # skip pages Tavily returned with no text
                continue
            doc = Document(
                page_content=extracted_page["raw_content"],
                metadata={"source": extracted_page["url"]},
            )
            all_pages.append(doc)
    log_success(f"✅ Extracted {len(all_pages)} pages from {len(all_pages)} pages successfully", Colors.GREEN)
    if failed_pages > 0:
        log_error(f"❌ Failed to extract {failed_pages} pages", Colors.RED)
    return all_pages

#=========================================================
# Index the chunks into the vector store
#=========================================================

async def index_documents_async(documents: List[Document], batch_size: int = 500):
    """Embed and add documents to the vector store in parallel batches."""
    log_header("VECTOR STORAGE PHASE")
    # Create Batches
    batches = [
        documents[i:i + batch_size] for i in range(0, len(documents), batch_size)
        ]

    log_info(
        f"📚 Indexing {len(documents)} chunks in {len(batches)} batches of up to {batch_size}",
        Colors.DARKCYAN,
    )
    #Process the batches concurrently
    async def add_batch(batch: List[Document], batch_num: int) -> bool:
        try:
            #
            await vectorstore.aadd_documents(batch)
            log_success(
                f"vestor store index successfully added the batches {batch_num}/{len(batches) } with {len(batch)} chunks")
            
        except Exception as e:
            log_error(f"Batch {batch_num} failed to index: {e}")
            return False
    return True

    #Process the batches concurrently
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    #count the number of successful batches
    succeeded = sum(1 for result in results if result is True)
    if succeeded == len(batches):
        log_success(f"All {len(batches)} batches indexed successfully")
    else:
        log_warning(f"{succeeded}/{len(batches)} batches indexed - {len(batches) - succeeded} failed")
    return succeeded == len(batches)



async def main():
  """Main async function to ingest data from the web."""
  log_header("DOCUMENTATION INGESTION PIPELINE")


  log_info(
        "🗺️  TavilyMap: Starting to map the documentation site : https://docs.langchain.com/",
        Colors.PURPLE,
    )

 #=========================================================
 # map the documentation site
 #=========================================================

  site_map = tavily_map.invoke( "https://docs.langchain.com/")
  #split then map into 20 batches
  url_batches = chunk_urls(list(site_map["results"]),chunk_size=20)

  log_success(
        f"🗺️  TavilyMap: Mapping completed successfully - {len(site_map['results'])} pages  URLs in {len(url_batches)} batches",
    )

  # Print every batch and the URLs inside it
  for batch_num, batch in enumerate(url_batches, start=1):
      log_info(f"📦 Batch {batch_num}/{len(url_batches)} - {len(batch)} URLs", Colors.CYAN)
      for url in batch:
          print(f"     - {url}")
#callingggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
  all_pages = await async_extract(url_batches)

  # Print one extracted page as a sample (first 1000 chars so the terminal isn't flooded)
  if all_pages:
      sample = all_pages[0]
      log_info(f"📄 Sample extracted page: {sample.metadata['source']}", Colors.CYAN)
      print(sample.page_content[:1000])
      print(f"... ({len(sample.page_content)} characters in total)")

 #=========================================================
 # Split documents into chunks
 #=========================================================
  log_header("DOCUMENT CHUNKING PHASE")
  log_info(
        f"✂️  Text Splitter: Processing {len(all_pages)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
  splitted_docs = text_splitter.split_documents(all_pages)
  log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_pages)} documents"
    )

  # Process documents asynchronously
  await index_documents_async(splitted_docs, batch_size=500)

  log_header("PIPELINE COMPLETE")
  log_success("🎉 Documentation ingestion pipeline finished successfully!")
  log_info("📊 Summary:", Colors.BOLD)
  log_info(f"   • Documents extracted: {len(all_pages)}")
  log_info(f"   • Chunks created: {len(splitted_docs)}")

 #=========================================================
 # extract the content from the documentation site
 #=========================================================
 


if __name__ == "__main__":
    asyncio.run(main())