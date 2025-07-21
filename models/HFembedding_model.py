from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings
load_dotenv()



class get_embedding_model:
    def __init__(self, repo_id: str, task: str, provider: str):
        self.repo_id = repo_id
        self.task = task
        self.provider = provider

        embedModel = HuggingFaceEndpointEmbeddings(
            model=self.repo_id,
            task=self.task,
            provider=self.provider
        )
        print("LOADED EMBEDDING MODEL: ", self.repo_id)
        self.model = embedModel

    def get_model(self):
        return self.model



# model = "sentence-transformers/all-mpnet-base-v2"