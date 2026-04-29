from langchain_community.document_loaders import PyPDFLoader

def load_pdf_pages(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load()   # one Document per page
