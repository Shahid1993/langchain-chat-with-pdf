from dotenv import load_dotenv
import os
from uuid import uuid4
import streamlit as st
from PyPDF2 import PdfReader
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.callbacks.manager import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
import faiss

AZURE_OPENAI_EMBEDDING_API_KEY = os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY")
AZURE_OPENAI_EMBEDDING_ENDPOINT = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
AZURE_OPENAI_EMBEDDING_API_VERSION= os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION")
os.environ["SSL_CERT_FILE"] = r"C:\Users\shahi\anaconda3\envs\streamlitdev\Library\ssl\cacert.pem"

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with your PDF")
    st.header("Chat with your PDF")

    # upload the PDF
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # extract the text
    if pdf is not None:
        reader = PdfReader(pdf)
        st.write(f"Number of pages: {len(reader.pages)}")

        text = ""
        for page in reader.pages:
            text += page.extract_text()

        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )

        chunks = text_splitter.split_text(text)

        # st.write(chunks)

        embeddings = AzureOpenAIEmbeddings(
            model="text-embedding-3-large",
            dimensions=3072, # Can specify dimensions with new text-embedding-3 models
            azure_endpoint=AZURE_OPENAI_EMBEDDING_ENDPOINT, #If not provided, will read env variable AZURE_OPENAI_ENDPOINT
            api_key=AZURE_OPENAI_EMBEDDING_API_KEY, # Can provide an API key directly. If missing read env variable AZURE_OPENAI_API_KEY
            openai_api_version=AZURE_OPENAI_EMBEDDING_API_VERSION, # If not provided, will read env variable AZURE_OPENAI_API_VERSION
        )

        
        # initialize the vector store
        index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )

        # index the chunks
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={}
            )
            documents.append(doc)

        uuids = [str(uuid4()) for _ in range(len(documents))]
        print(f"Total vectors: {len(documents)}")
        vector_store.add_documents(documents=documents, ids=uuids)
        llm = AzureChatOpenAI(
            azure_deployment="gpt-4o-mini"
        )

        chain = load_qa_chain(llm, chain_type="stuff")

        user_question = st.text_input(label="Ask a question about your PDF:")

        if user_question:
            docs = vector_store.similarity_search(user_question, k=5)
            with get_openai_callback() as cb:
                response = chain.run(question=user_question, input_documents=docs)
                print(cb)

            st.write(response)


if __name__ == '__main__':
    main()