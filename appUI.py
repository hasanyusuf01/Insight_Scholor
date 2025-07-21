import streamlit as st
from models import get_model, get_embedding_model
from loaders import docLoader
from Prompts import get_base_prompt
from spliters import get_text_splitters
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate
import os
import tempfile
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# Helper to format retrieved docs
def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

# UI
st.set_page_config(page_title="PDF QA", layout="centered")
st.title(" InSight Scholar")
st.markdown("Upload a PDF/Research Paper and ask any question about its content.")

st.markdown("**Select Model**")
col1, _ = st.columns([1, 5])
with col1:
    model_choice = st.selectbox(
        label="Model",
        options=["Kimi", "DeepSeek", "LLaMA"],
        index=0,
        label_visibility="collapsed"
    )

# File uploader

uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

if uploaded_file:
    # Save PDF to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        temp_pdf_path = tmp.name

    # Load PDF
    st.info("Loading and processing document...")
    doc = docLoader(file_path=temp_pdf_path)
    docs = doc.loadpdf()

    # Chunking
    splitter = get_text_splitters(chunk_size=500, chunk_overlap=0)
    chunks = splitter.split_documents(docs)

    # Embedding Model
    embedding_model = get_embedding_model(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction",
        provider="hf-inference"
    ).get_model()

    # Vector Store (using a temp directory to avoid conflicts)
    persist_dir = tempfile.mkdtemp()
    vector_store = Chroma(
        embedding_function=embedding_model,
        persist_directory=persist_dir,
        collection_name="runtime_pdf"
    )
    vector_store.add_documents(chunks)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})



##  un used code


    # class condition( BaseModel):
    #     task : Literal["code-generation", "text-generation"] = Field(description="The type of task asked to perform with the model.")
    # parser_1 =  PydanticOutputParser(pydantic_object=condition)

    # prompt_1 = PromptTemplate(
    #     template = """
    #     You are an expert classifier.

    #     Given the following user query, determine whether it is a `code-generation` or `text-generation` task.

    #     Respond ONLY in the following JSON format:
    #     {format_instructions}

    #     User Query: {question}
    #     """,
    #     input_variables=['question'],
    #     partial_variables= {'format_instructions': parser_1.get_format_instructions()}
    # )


    # Prompt Template
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided context.
        If the context is insufficient, just say you don't know.
        {context}
        Question: {question}
        """,
        input_variables=['context', 'question']
    )

    # Load model
    model = get_model(repo_id="moonshotai/Kimi-K2-Instruct", task="text-generation", provider="novita").get_model()
    codeModel = get_model(repo_id="deepseek-ai/DeepSeek-R1-0528", task="text-generation", provider="novita").get_model()
    llamaModel = get_model(repo_id="meta-llama/Llama-3.2-3B-Instruct", task="text-generation", provider="novita").get_model()

    parser = StrOutputParser()



    # Build Chain
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    # classify_chain = prompt_1 | model | parser_1
    main_chain = parallel_chain | prompt | model | parser

    # code_chain = parallel_chain | prompt | codeModel | parser

    # branch_chain = RunnableBranch(
    #     (lambda x:x.task == 'text-generation', main_chain),
    #     (lambda x:x.task == 'code-generation', code_chain),
    #     RunnableLambda(lambda x: "could not find task type, please try again.")
    # )

    # chain = classify_chain | branch_chain

    chain = main_chain
    model_chains = {
        "Kimi": parallel_chain | prompt | model | parser,
        "DeepSeek": parallel_chain | prompt | codeModel | parser,
        "LLaMA": parallel_chain | prompt | llamaModel | parser
    }
    
    # count = 0 
    # chat_history = [
    #     SystemMessage(content="You are a helpful Research assistant that answers questions based on the provided context."),

    # ]

    # Query Input
    user_query = st.text_input("Ask a question based on the uploaded PDF:")

    if user_query:
        # chat_history.append(HumanMessage(content=user_query))
        with st.spinner("Thinking..."):
            selected_chain = model_chains[model_choice]
            result = selected_chain.invoke(user_query)
            # chat_history.append(AIMessage(content=result))
            # count += 1
            st.success("Answer:")
            st.write(result)
