from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "gpt2-medium"
LORA_PATH = "error_aware_lora"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
model.config.use_cache = False

model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

prompt = """
You are a physics reasoning assistant.

Problem:
A resistor of 10 ohms carries a current of 2 A.

Student Answer:
V = R / I = 5 V

Task:
Explain why the student's answer is incorrect, state the correct formula, and compute the correct voltage.

### Answer:

"""

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_new_tokens=80,
    do_sample=False,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))