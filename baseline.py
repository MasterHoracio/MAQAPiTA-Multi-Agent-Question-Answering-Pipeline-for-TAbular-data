import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module='tqdm')
warnings.filterwarnings("ignore", category=UserWarning, module='tqdm.notebook')

import sys
sys.path.append("..")

import torch

import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from utilities import Generator
from utilities import promptGenerator

from databench_eval import Evaluator
from databench_eval.utils import load_qa, load_table

import os
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
    return """
        
        exec_string = (
            lead
            + response
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

def run_baseline(qa_dataset, gen):
    responses = []

    for instance in tqdm(qa_dataset):
        prompt_handler = promptGenerator.promptGenerator(instance, False)
        prompt         = prompt_handler.build_full_prompt(prompt_handler.baselinePrompt(instance), "baseline")
        code           = gen.getLLMOutput(prompt, 200, True, 0.8, 0.9)
        
        # RUN CODE
        output_code    = example_postprocess(code, instance)
        
        responses.append([str(output_code)])

def main():
    device     = torch.device("cuda:3")  # Select GPU 3
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct" # "google/gemma-3-4b-it"
    gen        = Generator.Generator(model_name, device)

    qa_dev     = load_qa(lang="ES", name="iberlef", split="dev")

    responses  = run_baseline(qa_dev, gen)

    evaluator  = Evaluator(qa=qa_dev)
    print(f"DataBenchSPA accuracy is {evaluator.eval(responses)}")
    save_responses(responses, "predictions.txt")

if __name__ == "__main__":
    main()