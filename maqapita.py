import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module='tqdm')
warnings.filterwarnings("ignore", category=UserWarning, module='tqdm.notebook')

import sys
sys.path.append("..")

import os
import ast
import json

import torch

import numpy as np
import pandas as pd
from tqdm import tqdm

from utilities import Generator
from utilities import promptGenerator

from databench_eval import Evaluator
from databench_eval.utils import load_qa, load_table

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def save_responses(responses, save_path: str) -> None:
    with open(save_path, "w") as f:
        for response in responses:
            f.write(str(response).replace("\n", " ") + "\n")

def example_postprocess(response: str, row: dict):
    try:
        df = load_table(row["dataset"], lang="ES")
        
        global ans
        lead = """
def answer(df):
    """
        exec_string = (
            response
            + "\nans = answer(df)"
        )
        local_vars = {"df": df, "pd": pd, "np": np}
        exec(exec_string, local_vars)

        ans = local_vars["ans"]
        if isinstance(ans, pd.Series):
            ans = ans.tolist()
        elif isinstance(ans, pd.DataFrame):
            ans = ans.iloc[:, 0].tolist()
        return ans.split("\n")[0] if "\n" in str(ans) else ans
    except Exception as e:
        return f"__CODE_ERROR__: {e}"

def run_pipeline(qa_dataset, gen):
    responses = []

    for instance in tqdm(qa_dataset):
        prompt_handler = promptGenerator.promptGenerator(instance, True)

        # GET COLUMNS
        prompt_columns = prompt_handler.build_full_prompt(prompt_handler.getColumns(), "columns")
        columns        = gen.getLLMOutput(prompt_columns, 100, True, 0.7, 0.8)
        prompt_handler.setReducedColumns(columns)
        
        # GET INSTRUCTIONS
        prompt_instructions = prompt_handler.build_full_prompt(prompt_handler.getInstructions(columns), "instructions")
        cot_instructions    = gen.getLLMOutput(prompt_instructions, 300, True, 0.7, 0.8)

        # GET CODE
        prompt_code = prompt_handler.build_full_prompt(prompt_handler.getCode(cot_instructions, columns), "code")
        code        = gen.getLLMOutput(prompt_code, 300, True, 0.8, 0.9)

        # RUN CODE
        output_code = example_postprocess(code, instance)

        # DEBUG CODE
        if str(output_code).startswith("__CODE_ERROR__"):
            attempts = 1
            while str(output_code).startswith("__CODE_ERROR__") and attempts <= 2:
                # GET "FIXED" CODE
                prompt_fix_bug = prompt_handler.build_full_prompt(prompt_handler.getCorrectCode(cot_instructions, columns, code, output_code), "code_correction")
                fixed_code     = gen.getLLMOutput(prompt_fix_bug, 400, True, 0.8, 0.9)
                output_code    = example_postprocess(fixed_code, instance)
                attempts += 1

        responses.append([str(output_code)])
    
def main():
    device     = torch.device("cuda:3")  # Select GPU 4
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct" # "google/gemma-3-4b-it"
    gen        = Generator.Generator(model_name, device)

    qa_dev = load_qa(lang="ES", name="iberlef", split="dev")
    responses = run_pipeline(qa_dev, gen)

    evaluator = Evaluator(qa=qa_dev)
    print(f"DataBenchSPA accuracy is {evaluator.eval(responses)}")
    save_responses(responses, "predictions.txt")

    
if __name__ == "__main__":
    main()