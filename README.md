# RAG-System
# RAG-system Implementation for "Leave No Context Behind” Paper

## Introduction
In this project, we implement a Retrieval-Augmented Generation (RAG) system utilizing the LangChain framework and the power of LLM (Large Language Model) like Gemini 1.5 Pro. The goal is to answer questions related to the "Leave No Context Behind" paper published by Google on April 10, 2024.

### : Generating and Storing Embeddings of Documents
This section involves preparing the documents and storing their embeddings for future use. It is executed in Visual Studio Code and involves the following steps:

- **Library Installation:** Install necessary libraries such as Pypdf, LangChain, NLTK, and Chroma.
- **Load PDF and Create Object:** Load the PDF of the "Leave No Context Behind" paper and create its object for processing.
- **Text Splitting:** Split the text using NLTKTextSplitter to segment it into manageable portions.
- **Chunking and Embedding:** Create chunks of text and generate embeddings using an embedding model.
- **Storage:** Store the generated embeddings in ChromaDB for persistence and future use.

### Integrating with Streamlit Using VS Code
This section involves integrating the RAG system with Streamlit for a user-friendly interface. The steps include:

- **Connection Setup:** Establish a connection with ChromaDB and convert it to a retriever object for efficient data retrieval.
- **Template Creation:** Develop a chat template for user interaction.
- **Model and Parser Creation:** Implement a chat model, output parser, and embedding model.
- **User Interaction:** Accept user input, perform queries on the database, and respond back to the user interface with relevant information.

## Resources
- **LinkedIn Profile:** [Reshma Shaik](https://www.linkedin.com/in/reshma-shaik-b38325245/)
