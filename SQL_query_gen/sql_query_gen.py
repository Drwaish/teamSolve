'''Core of sql query generation'''
from typing import List
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import utilities as ut


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
class Query:
    """
    A class use to generate response in sql.
    
    Attributes
    ----------
    Llm
        Llm for inference.

    Methods
    -------
    model_initialize
        Initialize the LLM model.
    
    query_generator
        Generate query according prompt.
    """
    def __init__(self):
        self.MODLE_NAME = "HuggingFaceH4/zephyr-7b-beta"
        logger.info("Initializing QueryProcessor")
        self.pipe = self.model_initialize()
        self.sys_prompt = ut.read_json('prompt.json')['system']
    def model_initialize(self):
        """
        Initialize the model.

        Parameters
        ----------
        None

        Return
        ------
        Llm 
            Llm model object. 
        """
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.MODLE_NAME,
                load_in_8bit=True,
                )
            tokenizer = AutoTokenizer.from_pretrained(self.MODLE_NAME)
            pipe = pipeline(
                "text-generation",
                    model=base_model,
                    tokenizer=tokenizer,
                    max_length=512,
                    temperature=0.1,
                    top_p=0.95,
                )
            logger.info("Pipeline Initialized")
            return pipe
        except Exception as e:
            print("Error in Model Initialization", e)
        
    def create_prompt(self,query:str, table : str,cols : List[str])->str:
        """
        Create prompt according to query.

        Parameters
        ----------
        query
            Query user want to convert into sql.
        table
            Table from which query will be executed.
        cols
            Columns in the table
        
        Return
        prompt
            Prompt to execute for inference.

        """
        try:
            messages = [
                    {
                        "role": "system",
                        "content": self.sys_prompt,
                    },
                    {"role": "user",
                    "content": query + 'from this tabel '+ table +' using these columns '+ cols},
                ]
            prompt = self.pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            logger.info("Prompt Created")
            return prompt
        except Exception as e:
            print("Error in prompt creation", e)
            return "Error in prompt"

    def create_prompt_multiple_table(self, query:str, table : List[str],cols : List[List[str]])->str:
        """
        Create prompt according to query.

        Parameters
        ----------
        query
            Query user want to convert into sql.
        tables
            List of Tables from which query will be executed.
        cols
            List of Columns in the table
        
        Return
        prompt
            Prompt to execute for inference.

        """
        # template = """Generate a SQL query using the following table name: {Table}, and columns as a list: {Columns}, to answer the following question:
        # {query}.
        # Output Query:

        # """
        try:
            messages = [
                    {
                        "role": "system",
                        "content": self.sys_prompt,
                    },
                    {"role": "user",
                    "content": query + 'from this tabel '+ str(table) +' using these columns '+ str(cols)},
                ]
            prompt = self.pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            logger.info("Prompt Created")
            return prompt
        except Exception as e:
            print("Error in prompt creation", e)
            return "Error in prompt"

    def query_generator(self,prompt: str):
        """
        Generate response using prompt.
        
        Parameters
        ----------
        prompt
            Query and instruction to LLM to generate queries.

        """
        try:   
            outputs = self.pipe(prompt)
            if outputs:
                logger.info("Response Created")
                return outputs[0]['generated_text'].split('<|assistant|>')
        except Exception as e:
           logger.info("Error in output creation "+ e)

