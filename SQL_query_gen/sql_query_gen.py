'''Core of sql query generation'''
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import utilities as ut

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
        return pipe
        
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
        # template = """Generate a SQL query using the following table name: {Table}, and columns as a list: {Columns}, to answer the following question:
        # {query}.
        # Output Query:

        # """
        messages = [
                {
                    "role": "system",
                    "content": self.sys_prompt,
                },
                {"role": "user",
                "content": query + 'from this tabel '+ table +' using these columns '+ cols},
            ]
        prompt = self.pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    def query_generator(self,prompt):
        """
        Generate response using prompt.
        
        Parameters
        ----------
        prompt

        """
        outputs = self.pipe(prompt, max_new_tokens=512)
        return outputs[0]['generated_text']

if __name__=="__main__":
    transaction = ["EmployeeID", "FirstName,", "LastName" ,"Age" ,"Department", "Position","Salary","HireDate","ManagerID"]
    table = "transaction"
    query = "Tell me highest salary in engineering department."
    qu =  Query()
    prompt = qu.create_prompt(table=table, cols= transaction, query=query)
    print(qu.query_generator(prompt=prompt))