from databench_eval.utils import load_table
import pandas as pd

class promptGenerator():
    def __init__(self, row: dict, FewShot: bool):
        super(promptGenerator, self).__init__()
        self.dataset            = row["dataset"]
        self.question           = row["question"]
        self.type_ans           = row["type"]
        self.df                 = load_table(row["dataset"], lang="ES")
        self.python_types       = self.df.dtypes.apply(lambda x: x.type)
        self.columns            = list(self.df.columns)

        self.agent_headers      = {
        "code": "You are a numpy and pandas code generator. Complete the function provided.",
        "instructions": "You are an experienced software engineer. Describe the high-level steps to address and solve a given question.",
        "columns": "You are an experienced data analyst. Select the column(s) of a DataFrame needed to answer a given question.",
        "code_correction": "You are an experienced Python developer. Fix the following bug.",
        "default": "You are a helpful assistant, please follow every instruction in the following prompt."
        }
        
        if FewShot:
            with open('utilities/few_shot_samples.txt', 'r', encoding='utf-8') as file:
                content = file.read()
            self.samples = content
        else:
            self.samples = None

    def build_full_prompt(self, prompt, type_prompt):
        system_role = {"role": "system", "content": self.agent_headers[type_prompt]}
        full_prompt = [system_role, {"role": "user", "content": prompt}]
        return full_prompt
    
    def getColumnNames(self, list_indices):
        return [self.df.columns[i] for i in list_indices]

    def setReducedColumns(self, columns):
        reduced = []
        
        for column in columns:
            if column in self.df.columns:
                reduced.append(column)
        
        self.reduced_columns = reduced

    def getColumns(self) -> str:
        return f'''
                From these columns: {self.df.columns.tolist()}, select ONLY the column(s) needed to answer the following question:  
                "{self.question}"
                
                Details:
                - Expected answer type: {self.type_ans}  
                - Column types: {self.python_types}  
                - First 5 rows:  
                {self.df.head(5).to_string()}
                
                Instructions:  
                * Consider column data types before selecting.
                * Use EXACT column names shown above.  
                * Reply with a Python list of strings—only the selected column(s).  
                * NOTHING else—no extra text or formatting.
                * Choose the FEWEST columns needed (1 – 2 max).  
                '''
    
    def getInstructions(self, relevantColumns: str) -> str:
        return f'''
        You have a pandas DataFrame `df` with:
        df.columns = {self.reduced_columns}
        Column types: {self.df[self.reduced_columns].dtypes.apply(lambda x: x.type)}

        Question:
        "{self.question}"

        Details:
        - Expected answer type: {self.type_ans}
        - Relevant columns: {relevantColumns}
        - First 10 rows:
        {self.df[self.reduced_columns].head(10).to_string()}

        Instructions:
        * Describe high-level steps to answer the question using only the DataFrame.
        * Be abstract—no specific values or examples.
        * Use EXACT column names.
        * DO NOT write code.
        * Keep it brief and focused.
        '''
    
    def getCode(self, Steps: str, relevantColumns: str) -> str:
        col_values = ""

        for col in relevantColumns:
            col_values += (col + ": ")
            if col in self.df.columns:
                valores_cols += ", ".join(self.df[col].unique().tolist())

        return f'''
        Create and complete the function below to answer the following question: "{self.question}", using ONLY the DataFrame information.

        Context:
        - Input: pandas DataFrame `df`
        - Columns: {self.reduced_columns}
        - Types: {self.df[self.reduced_columns].dtypes.apply(lambda x: x.type)}
        - Expected answer type: {self.type_ans}
        - Relevant columns: {relevantColumns}
        - The different values of the columns to answer the question are: {col_values}
        - Suggested steps: {Steps}
        - First 5 rows:
        {self.df[self.reduced_columns].head(5).to_string()}

        Instructions:
        * DO NOT wrap the code in ```python``` or any other code block—plain text only.  
        * Output ONLY the function—no extra text or formatting.
        * Use EXACT column names as shown.
        * Use ONLY pandas, numpy, and built-in Python.
        * Keep type formatting correct (e.g., return `True`, not `np.True_`).
        * Add a comment before each line explaining the logic.
        * No file reading or external sources.
        * Be concise and follow the format.

        Possible outputs:
        1. `True` or `False`
        2. A single value (e.g., category or number)
        3. A list of values

        Few-shot examples:
        {self.samples}

        Now complete this function:

        import pandas as pd
        import numpy as np

        # Function to complete, start your answer here
        def answer(df: pd.DataFrame):
            """Explain each step with comments above."""
            df.columns = {self.reduced_columns}
        '''

    def getCorrectCode(self, Steps: str, relevantColumns: str, code: str, error: str) -> str:
        col_values = ""

        for col in relevantColumns:
            col_values += (col + ": ")
            if col in self.df.columns:
                valores_cols += ", ".join(self.df[col].unique().tolist())
        
        return f'''
        Fix the following function so it correctly answers:
        "{self.question}"
        using ONLY the provided DataFrame `df`.
        
        Context:
        - df.columns = {self.reduced_columns}
        - Column types: {self.df[self.reduced_columns].dtypes.apply(lambda x: x.type)}
        - Expected output: {self.type_ans}
        - Use columns: {relevantColumns}
        - Possible values: {col_values}
        - Suggested steps: {Steps}
        - First 5 rows:
        {self.df[self.reduced_columns].head(5).to_string()}
        
        Current (incorrect) function:
        {code}
        
        Error:
        {error}
        
        Instructions:
        * DO NOT wrap the code in ```python``` or any other code block—plain text only.  
        * Provide ONLY the corrected code inside the function—no extra text.
        * Use exact column names and only pandas, numpy, and built-in functions.
        * Output format must match {self.type_ans}.
        * Comment each line in the function to explain it.
        * Do not read files or use external sources.
        
        Valid answer formats:
        1. `True` or `False`
        2. A single value (number or category)
        3. A list of values
        
        Rewrite below:
        
        import pandas as pd
        import numpy as np

        # Function to correct, start your answer here
        def answer(df: pd.DataFrame):
            """Only write code inside this function with line-by-line comments."""
            df.columns = {self.reduced_columns}
        '''