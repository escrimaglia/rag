from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import AzureChatOpenAI, OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from dotenv import load_dotenv
from pathlib import Path


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

    system_prompt = (
        "You are a helpful assistant for question-answering tasks."
        "Your goal is to provide accurate and concise answers to user queries."
        "Use the following pieces of retrieved context to answer questions."
        "If you don't know the answer, just say you don't know. Don't try to make up an answer."
        "Use three sentences maximun and keep the answer concise. The answer should always be in Spanish, translate if necessary."
        "\n"
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("user", "{input}"),
        ("system", system_prompt)
    ])

    llm = AzureChatOpenAI(model="gpt-4.1-mini", temperature=0.2, max_tokens=4000, api_version="2024-10-21")

    answer = create_stuff_documents_chain(llm, prompt=prompt)
    rag_chain = create_retrieval_chain(retriever, answer)
    response = rag_chain.invoke({"input": """Calcular máximos y mínimos relativos, si existen, de las siguientes funciones:
                                    a) y = 2x - 3x - x^2 
                                    b) y = x^2 - 12x
                                    c) y = 3x + 12/x + 20
                                 """})

    composed_context = format_docs(response.get("context", []))
    answer = response.get("answer", "no answer")
    
    references = ""
    for line in response.get("context", []):
        source = line.metadata.get('source', 'unknown source').split('/')[-1]
        references += (f"{'\n'} -> Page: {line.metadata.get('page', 0)}, Source: {source if source else 'unknown source'}")

    print (f"{'\n'}References: {references}")
    print (f"{'\n'}Answer: {'\n'}{answer}")

if __name__ == "__main__":
    main()
