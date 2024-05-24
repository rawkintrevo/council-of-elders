from councilofelders.agent import Agent
from councilofelders.utils import merge_items_by_role, update_role

from llama_index.llms.openai import OpenAI
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.llms import ChatMessage

class LlamaIndexOpenAIAgent(Agent):
    def __init__(self,
                 model,
                 system_prompt,
                 temperature,
                 name,
                 openai_api_key,
                 pinecone_index_name,
                 pinecone_api_key,
                 top_k):
        llm = OpenAI(model=model,
                     temperature=temperature,
                     system_prompt=system_prompt,
                     reuse_client=False,
                     api_key= openai_api_key)
        pc = Pinecone(api_key=pinecone_api_key, environment="gcp-starter")
        pinecone_index = pc.Index(pinecone_index_name)
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        embed_model = OpenAIEmbedding(api_key=openai_api_key)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store,
                                                   embed_model=embed_model)
        super().__init__(index.as_chat_engine(
            chat_mode="context",
            llm=llm,
            similarity_top_k=top_k,
        ),
            model,
            temperature,
            name)


    def add_message_to_history(self, msg, who):
        if (who != 'user') or (who != 'system'):
            who = 'assistant'
        self.history.append({'content': msg, 'role': who})

    def generate_next_message(self):
        query = self.history[-1]['content']
        response = self.client.chat(query,
                                    chat_history=
                                    self.format_message_list_for_llama_index(
                                        merge_items_by_role(
                                            update_role(self.history, self.name)
                                        )
                                    )
                    )
        self.sources = [{'url': source_node.node.metadata.get('src', ''),
                    'title': source_node.node.metadata.get('title', '')} for
                   source_node in response.source_nodes]
        return response.response

    def format_message_list_for_llama_index(self, messages):
        return [ChatMessage(role= message["role"],
                            content= message["content"]) for message in messages]
