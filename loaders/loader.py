from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, ArxivLoader



class docLoader:
    def __init__(self, file_path: str):
        self.loader = PyPDFLoader(file_path)
        print("LOADED PDF: ", file_path)

    def loadpdf(self):
        return self.loader.load()
    
    def get_pdf_info(self):
        docs = self.loadpdf()
        print(len(docs))
        print(docs)


class textLoader:
    def __init__(self, file_path: str):
        self.loader = TextLoader(file_path, encoding='utf-8')

    def loadtext(self):
        return self.loader.load()

class csvLoader:
    def __init__(self, file_path: str):
        self.loader = CSVLoader(file_path)

    def loadcsv(self):
        return self.loader.load()

class arxivLoader:
    def __init__(self, query: str):
        self.loader = ArxivLoader(query=query)

    def loadarxiv(self):
        return self.loader.load()

# print(docs[1].metadata)