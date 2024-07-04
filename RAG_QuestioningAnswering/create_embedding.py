''' Create embeddings from file '''
import os
import logging
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Embeddings:
    """
    Load document and create embeddding.

    Attributes
    ----------
    file_name
        Name of the file need to create embeddings.


    """
    def __init__(self):
        logger.info("Call Embedding Constructor ")
        self.repo_name = "HuggingFaceH4/zephyr-7b-beta"
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        # self.embeddings = HuggingFaceEmbeddings(self.embedding_model,model_kwargs={'device': 'cuda'})
        # self.document = self.load_document()
        self.llm = self.llm_initialization()
        # self.retriever = self.store_vectordb("RaGQADoc.pdf")
        self.text_splitter=RecursiveCharacterTextSplitter(chunk_size=100,
                                                          chunk_overlap=20)
    
    def load_document(self, file_name:str = None):
        """
        Load pdf document.

        Parameters
        ----------
        None

        Return
        Unstructured document object
        """
        try:
            loader = UnstructuredFileLoader(file_name)
            documents = loader.load()
            logger.info("Load Document")
            return documents
        except Exception as e:
            logger.error(" "+str(e))

    def llm_initialization(self):
        """
        LLM initalization for inference.

        Parameters
        ----------
        None

        Return
        ------
        LLM Object
        """
        try:
            llm = HuggingFaceEndpoint(
                repo_id= self.repo_name,
                task="text-generation",
                max_new_tokens=1024,
                top_k= 5,
                temperature = 0.1,
                repetition_penalty = 1.03,
                huggingfacehub_api_token =  os.environ.get("HF_TOKEN") # Replace with your actual huggingface token
                )
            logger.info("LLM initialized")
            return llm
        except Exception as e:
            logger.error(" "+str(e))
    def update_db(self, file_path : str = None):
        """
        Update the vector store.

        Parameters
        ----------
        file_path
            Path of the file for update the vector store.

        Return
        ------
        None
        """
        try:
            docs = self.load_document(file_name=file_path)
            if docs is None:
              print("Error: No docs")
            text_chunks=self.text_splitter.split_documents(docs)
            embeddings = HuggingFaceEmbeddings(model_name = self.embedding_model,model_kwargs={'device': 'cuda'})

            vectordb = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
            vectordb.add_documents(text_chunks)
            retriever = vectordb.as_retriever()
            logger.info("Vector DB Updated")
            return retriever
        except Exception as e:
            logger.error(" "+str(e))

    def store_vectordb(self, file_path_name):
        """
        Update the vector store.

        Parameters
        ----------
        None

        Return
        ------
        None
        """
        try:
            docs = self.load_document(file_name=file_path_name)
            if docs is None:
              print("Error: No Docs")
            embeddings = HuggingFaceEmbeddings(model_name = self.embedding_model, model_kwargs={'device': 'cuda'})
            text_chunks=self.text_splitter.split_documents(docs)
            vectorstore = Chroma.from_documents(documents=text_chunks, embedding=embeddings, persist_directory = './chroma_db')
            retriever = vectorstore.as_retriever()
            logger.info("Create Vector DB")
            return retriever
        except Exception as e:
            logger.error(" "+str(e))
