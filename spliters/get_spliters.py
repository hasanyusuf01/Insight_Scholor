from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

def get_text_splitters(chunk_size=200, chunk_overlap=0):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )


def get_code_splitters(chunk_size=200, chunk_overlap=0, lang=str):
    spliter = RecursiveCharacterTextSplitter.from_language(
        language=Language.from_string(lang),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return spliter