import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

if __name__ == "__main__":
    print("Ingesting....")

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.environ["INDEX_NAME"]

    # Skip creation on re-runs -- create_index fails if the index already exists
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=1536,            # must match your embedding model
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),  # free-plan region
        )
        print(f"Created index: {index_name}")
    else:
        print(f"Index already exists: {index_name}")
