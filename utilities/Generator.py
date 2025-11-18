import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module='tqdm')
warnings.filterwarnings("ignore", category=UserWarning, module='tqdm.notebook')

import transformers
import torch

class Generator:
    def __init__(self, model: str, device: object):
        self.device         = device
        self.model_name     = model

        display_msg = "Loading: " + self.model_name
        print("#"*70)
        print(display_msg.center(70))
        print("#"*70)
        
        self.pipeline       = transformers.pipeline(
                                    "text-generation",
                                    model=self.model_name,
                                    model_kwargs={"torch_dtype": torch.bfloat16},
                                    device_map=device,
                                )
        torch.cuda.set_device(device)

    def getLLMOutput(self, prompt: str, max_tokens: int, do_sample: bool, temperature: float, top_p: float) -> str:
        output = self.pipeline(
            prompt,
            max_new_tokens = max_tokens,
            pad_token_id   = self.pipeline.tokenizer.eos_token_id,
            do_sample      = do_sample,
            temperature    = temperature,
            top_p          = top_p
        )

        output = output[0]["generated_text"][-1]['content']

        return output