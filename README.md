# Retrieval-Augmented Generation (RAG)

## RAG with PDFs, OpenIA Embedding and AzureOpenAI LLM

### RAG Using LCEL

This project demonstrates a simple Retrieval-Augmented Generation (RAG) pipeline using **LangChain**, **Chroma**, and **Azure OpenAI**

### How RAG LCEL works

1. **Load PDFs**: Documents inside the `./pdfs` directory are read using `PyPDFDirectoryLoader`.  
2. **Split Text**: Each document is divided into overlapping chunks with `RecursiveCharacterTextSplitter`.  
3. **Vector Store**: Chunks are embedded with `OpenAIEmbeddings` and stored in a Chroma vector database.  
4. **Retriever**: The vector store acts as a retriever to fetch relevant chunks for user queries.  
5. **LLM**: The script uses Azure OpenAI (`gpt-4.1-mini`) with controlled parameters (temperature, token limit).  
6. **Prompt**: A RAG prompt is pulled from LangChain Hub to structure responses.  
7. **Chain**: Query → Retriever → Prompt → LLM → Answer.  
8. **Output**: The final answer is generated based on the PDF content.  

### Run RAG LCEL

```bash
python rag_lcel.py
```

### RAG Langchain

This project implements a Retrieval-Augmented Generation (RAG) pipeline using **LangChain**, **Chroma**, and **Azure OpenAI** with a custom prompt design.  

### How RAG Langchain works

1. **Load PDFs**: Documents are loaded from the `./pdfs` folder with `PyPDFDirectoryLoader`.  
2. **Split Text**: Each PDF is divided into overlapping chunks using `RecursiveCharacterTextSplitter`.  
3. **Vector Store**: Chunks are embedded via `OpenAIEmbeddings` and stored in a Chroma database.  
4. **Retriever**: The vector store acts as a retriever to fetch relevant document chunks for user queries.  
5. **Prompt Template**: A `ChatPromptTemplate` defines the assistant’s behavior. It always answers in Spanish, keeps answers concise, and admits when it does not know.  
6. **LLM**: Azure OpenAI (`gpt-4.1-mini`) is used as the language model for generating answers.  
7. **RAG Chain**: `create_stuff_documents_chain` builds the answer chain, and `create_retrieval_chain` connects it to the retriever.  
8. **Execution**: The script queries the model, retrieves context, generates an answer, and prints both the references (page and source) and the final response.  

### Run RAG Langchain

```bash
python rag_langchain.py
```

### Requirements

- `Python 3.10+`
- `langchain`, `langchain-community`  
- `langchain-chroma`  
- `langchain-openai`  
- `python-dotenv`  
- `chromadb`

### Environment Variables

The following environment variables must be defined

- ```AZURE_OPENAI_ENDPOINT``` (if AzureOpenAI LLM is used)
- ```ZURE_OPENAI_API_KEY``` (if AzureOpenAI LLM is used)
- ```OPENAI_API_KEY``` (if OpenAI LLM and/or OpenAIEmbeddings are used)
- ```ANGCHAIN_API_KEY``` (if LangChain Hub for pulling prompts is used)

```Ed Scrimaglia```
  