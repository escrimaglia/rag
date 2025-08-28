from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import AzureChatOpenAI, OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from pathlib import Path
from langchain import hub

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    load_dotenv()
    pdf_dir = Path(__file__).parent / "./pdfs"
    document = PyPDFDirectoryLoader(pdf_dir).load()
    print(f"Documentos cargados: {len(document)}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(document)
    print(f"Chunks generados: {len(split_docs)}")
    for doc in split_docs:
        if hasattr(doc, 'page_content'):
            doc.page_content = doc.page_content.encode('utf-8', 'ignore').decode('utf-8')

    vectorstore = Chroma.from_documents(documents=split_docs, embedding=OpenAIEmbeddings(model="text-embedding-3-small"))
    retriever = vectorstore.as_retriever()
    llm = AzureChatOpenAI(model="gpt-4.1-mini", temperature=0.2, max_tokens=4000, api_version="2024-10-21")
    
    prompt = hub.pull("rlm/rag-prompt")

    lcel_chain = ({ "context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = lcel_chain.invoke("que son las inecuaciones en programacion lineal?, dame ejemplos")

    print (f"{'\n'}Answer: {'\n'}{response}")

if __name__ == "__main__":
    main()
