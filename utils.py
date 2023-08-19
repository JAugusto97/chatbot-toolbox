import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer

class llama2_orca_13b:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("OpenAssistant/llama2-13b-orca-8k-3319", trust_remote_code=True, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            "OpenAssistant/llama2-13b-orca-8k-3319",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="cuda",
            trust_remote_code=True
        )
        self.system_context = \
        """You are a helpful, respectful and honest assistant.
        Always answer as helpfully as possible, while being safe.
        Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
        Please ensure that your responses are socially unbiased and positive in nature.
        If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.
        If you don't know the answer to a question, please don't share false information.
        """

    def prompt(
        self,
        user_message,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        max_new_tokens=256
    ):
        prompt = f"""<|system|>{self.system_context}</s><|prompter|>{user_message}</s><|assistant|>"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        output = self.model.generate(
            **inputs,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
        )
        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        clean_output = decoded_output[len(self.system_contex)+len(user_message)+1:]
        return clean_output


class falcon_7b:
    def __init__(self):
        self.model = "OpenAssistant/falcon-7b-sft-mix-2000"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="cuda",
        )

    def prompt(self, user_message, max_new_tokens=256, do_sample=True, top_k=50, num_return_sequences=1):
        input_text = f"<|prompter|>{user_message}<|endoftext|><|assistant|>"
        sequences = self.pipeline(
            input_text,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            return_full_text=True,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id
        )
        ans = "".join([seq["generated_text"] for seq in sequences])
        ans = ans.split("<|assistant|>")[1]
        return ans

class falcon_40b:
    def __init__(self):
        self.model = "OpenAssistant/falcon-40b-sft-mix-1226"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="cuda",
        )

    def prompt(self, user_message, max_new_tokens=256, do_sample=True, top_k=50, num_return_sequences=1):
        input_text = f"<|prompter|>{user_message}<|endoftext|><|assistant|>"
        sequences = self.pipeline(
            input_text,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            return_full_text=True,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id
        )
        ans = "".join([seq["generated_text"] for seq in sequences])
        ans = ans.split("<|assistant|>")[1]
        return ans
    
class oasst_pythia_12b():
    def _init_(self):
        self.model = "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="cuda",
        )

    def prompt(self, user_message, max_new_tokens=256, do_sample=True, top_k=50, num_return_sequences=1):
        input_text = f"<|prompter|>{user_message}<|endoftext|><|assistant|>"
        sequences = self.pipeline(
            input_text,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            return_full_text=True,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id
        )
        ans = "".join([seq["generated_text"] for seq in sequences])
        ans = ans.split("<|assistant|>")[1]
        return ans
    
class llama_v1_30b_instruct():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("upstage/llama-30b-instruct-2048")
        self.model = AutoModelForCausalLM.from_pretrained(
            "upstage/llama-30b-instruct-2048",
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_8bit=True,
            rope_scaling={"type": "dynamic", "factor": 2} # allows handling of longer inputs
        )

    def prompt(self, user_message, max_new_tokens, do_sample=True, top_k=50):
        formatted_prompt = f"### User:\n{user_message}\n\n### Assistant:\n"
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        output = self.model.generate(
            **inputs,
            do_sample=do_sample,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
        )
        output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        output_text = output_text.split("### Assistant:\n")[1]
        return output_text
    
class llama_v2_13b_chat_hf():
    def __init__(self):
        model_name = "meta-llama/Llama-2-13b-chat-hf"
       
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipeline = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.float16,
            device_map="auto",

        )

    def prompt(self, user_message, temperature=0.1, max_new_tokens=256, do_sample=True, top_k=50, system_context=None):
        if system_context is None:
            system_context = "<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n<</SYS>>\n\n"
        else:
            system_context = f"<s>[INST] <<SYS>>\n{system_context}\n<</SYS>>\n\n"
        prompt = f"{system_context} {user_message} [/INST] "
        sequences = self.pipeline(
            prompt,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            return_full_text=True,
            top_k=top_k,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id
        )

        ans = "".join([seq["generated_text"] for seq in sequences])
        ans = ans.split("[/INST]")[1].strip()
        return ans