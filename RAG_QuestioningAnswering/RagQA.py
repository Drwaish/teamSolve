'''Question Answering using RAG'''
import logging
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import utilities as ut
from create_embedding import Embeddings



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Rag:
    def __init__(self):
        logger.info("Constructor Calling")
        self.sys_prompt = ut.read_json('prompt.json')['system']
        self.emb = Embeddings()
        self.emb.store_vectordb('RagQADoc.pdf')  # Create presisten DB
        self.emb.update_db('Resume_MirzaZainAliNasir.pdf')  # Update DB
    def create_prompt(self):
        """
        Create prompt accordint to query.

        Parameters
        ----------
        input
            Query by the user.
        
        Return
        prompt
          Object of chatprompttemplate for inference.
        """
        try:
            prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.sys_prompt),
                ("human", "{input}"),
            ]
            )   
            logger.info("Prompt template creation")
            return prompt
        except Exception as e:
            logger.info(""+e)
    def inference(self, input: str ):
        """
        Get response from LLM.

        Parameters
        ----------
        input
            Question asked by user

        Return
        ------
        Response by LLM using RAG.
        """
        try:
            prompt = self.create_prompt()
            question_answer_chain = create_stuff_documents_chain(self.emb.llm, prompt)
            rag_chain = create_retrieval_chain(self.emb.retriever, question_answer_chain)
            results = rag_chain.invoke({"input": input})
            logger.info("Inference Completed")
            return results
        except Exception as e:
            logger.info(""+e)
    
