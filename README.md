# teamSolve

Bot task are completed with extra requirements in their respective folders.

Clone the github repo from following command

```bash
git clone https://github.com/Drwaish/teamSolve.git
cd teamSolve
```

Now install the requirements using this command
```python
pip install -r requirements.txt
```

# SQL query generation

In this task I try with single and double tables and it responds accordingly.

If you want to run this task

```bash
cd SQL_query_gen
```
After this execute the **main.py** file.
```bash
python main.py
```

In this file, example of 2 tables are implemented. 
You need to specify the columns names and table names in input and it works accordingly.

Uncomment line 11 in **main.py** if you want to run on single table.

This will also works fine on large databases.

# RAG
If you want to run this task
```bash
cd RAG_QuestioningAnswering
```

After this run following command.
```bash
python main.py
```

I update the vector at runtime with my resume.First code will create vector db of inout doc

and then update the vector db  with resume and respond according to my resume.

I also try different inout question it respond accurately.

In this tasks I use zephyr 7b beta for inference.

### Thanks for your time.



