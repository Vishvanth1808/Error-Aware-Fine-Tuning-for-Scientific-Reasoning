from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# =============================
# CONFIG
# =============================
MODEL_NAME = "gpt2-medium"
DATA_FILE = "formatted_dataset.jsonl"
OUTPUT_DIR = "./error_aware_lora"

# =============================
# TOKENIZER
# =============================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# =============================
# LOAD DATASET
# =============================
dataset = load_dataset("json", data_files=DATA_FILE)

# =============================
# TOKENIZATION (CPU OPTIMIZED)
# =============================
def tokenize(batch):
    tokens = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=192   # reduced for CPU efficiency
    )

    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized = dataset.map(
    tokenize,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# =============================
# LOAD MODEL
# =============================
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Disable cache during training
model.config.use_cache = False

# Freeze token embeddings (CPU efficiency)
for param in model.get_input_embeddings().parameters():
    param.requires_grad = False

# =============================
# LoRA CONFIG
# =============================
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =============================
# TRAINING
# =============================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_train_epochs=2,
    learning_rate=2e-4,
    weight_decay=0.01,
    logging_steps=5,
    save_strategy="epoch",
    save_total_limit=1,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"]
)

# =============================
# TRAIN
# =============================
trainer.train()

# =============================
# SAVE ADAPTER
# =============================
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nTraining complete. LoRA adapters saved to", OUTPUT_DIR)