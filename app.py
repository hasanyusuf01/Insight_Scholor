from  models import get_model, get_embedding_model
from loaders import docLoader
from Prompts import get_base_prompt
from spliters import get_text_splitters
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

# initialize and load the model 
model = get_model(repo_id="moonshotai/Kimi-K2-Instruct", task="text-generation", provider="novita").get_model()

# load the prompt template
template = get_base_prompt().get_prompt()
# fill the values of the placeholders

# load the document
doc = docLoader(file_path="assets/JD.pdf")
docs = doc.loadpdf()

# generate the prompt using the template
# prompt = template.invoke({
#         "doc_input": docs[0].page_content,
#         "length_input": "detailed",
#         "queries": ["What is the job title?", "What are the required skills?"],
#         "style_input": "detailed"
# })
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)
# Initialize the output parser
parser = StrOutputParser()

# Initialize the splitter
splitter = get_text_splitters(chunk_size=500, chunk_overlap=0)
chunks = splitter.split_documents(docs)


# Initialize the embedding model
embeddingModel = get_embedding_model(repo_id="sentence-transformers/all-MiniLM-L6-v2", task="feature-extraction", provider="hf-inference").get_model()

# Initialize the vector store
vector_store = Chroma(
    embedding_function= embeddingModel,
    persist_directory='my_chroma_db',
    collection_name='sample'
)
vector_store.add_documents(chunks)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})



parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

main_chain = parallel_chain | prompt | model | parser

result = main_chain.invoke('give me the job description')

print(result)
