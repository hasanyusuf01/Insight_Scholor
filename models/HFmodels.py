from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()


class get_model:

    def __init__(self, repo_id: str, task: str, provider: str ):
        self.repo_id = repo_id
        self.task = task
        self.provider = provider

        llm = HuggingFaceEndpoint(
            repo_id=self.repo_id,
            task=self.task,
            provider=self.provider
        )
        model = ChatHuggingFace(llm=llm)
        self.model = model
        print("LOADED MODEL: ", self.repo_id)

    def get_model(self):
        return self.model
    
        # result = model.invoke("Write me a 5 line poem about the sea.")

        # print(result.content)