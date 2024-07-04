from sql_query_gen import Query


if __name__=="__main__":
    transaction = ["EmployeeID", "FirstName,", "LastName" ,"Age" ,"Department", "Position","Salary","HireDate","ManagerID"]
    transaction1 = ["EmployeeID", "InsuranceFirm", "InsuranceClaim"]
    tables = ["transaction", "transaction1"]
    query = "Tell me FirstName of the employe with highest InsuranceClaim"
    qu =  Query()
    # prompt = qu.create_prompt(table=table, cols= transaction, query=query)
    prompt = qu.create_prompt_multiple_table(table=tables,  cols= [transaction, transaction1], query=query)
    print(qu.query_generator(prompt=prompt))