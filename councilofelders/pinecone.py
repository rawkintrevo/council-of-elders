from pinecone import Pinecone, ServerlessSpec
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext

from llama_index.embeddings.openai import OpenAIEmbedding



class PineconeHelper:
    def __init__(self, pinecone_api_key, openai_api_key):
        self.embed_model = OpenAIEmbedding(api_key=openai_api_key)
        self.pc = Pinecone(api_key=pinecone_api_key)

    def create_index(self, name):
        self.pc.create_index(name=name,
                dimension=1536,
                metric="euclidean",
                 spec=ServerlessSpec(cloud="aws", region="us-west-2"),
        )

    def delete_index(self, name):
        self.pc.delete_index(name=name)

    def add_document(self, name: str, document: str):
        pinecone_index = self.pc.Index(name)
        file_metadata = lambda x: {"filename": x} # other metadat would go here
        documents = SimpleDirectoryReader(document, file_metadata=
                file_metadata).load_data()
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=self.embed_model
        )

