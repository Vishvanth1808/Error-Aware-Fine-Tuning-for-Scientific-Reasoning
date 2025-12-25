Error-Aware Fine-Tuning for Scientific Reasoning
Overview

This project explores error-aware fine-tuning of large language models for physics and scientific reasoning. Instead of training a model only on correct solutions, the system is trained on incorrect student answers paired with explanations of why they are wrong and how to correct them.

The goal of the project is not to build a perfect physics solver. Instead, it studies how parameter-efficient fine-tuning (LoRA) biases a language model’s behavior toward recognizing mistakes and attempting corrective explanations, and where such approaches fail.

The project emphasizes model behavior analysis and limitations, rather than accuracy alone.

Core Idea

Most fine-tuning approaches train models on correct answers only. In contrast, this project explicitly models student errors.

Each training example contains:

A physics problem

A student’s incorrect answer

The type of error made

An explanation of why the answer is wrong

A strategy for correction

The correct explanation

By exposing the model to these patterns, the project investigates whether the model learns to:

Identify that an answer is wrong

Attempt a correction

Produce structured explanatory responses

Error Taxonomy

The project defines a small but meaningful taxonomy of error types, such as conceptual errors, formula misuse, calculation errors, unit errors, sign errors, and assumption errors.

These error labels are included directly in the training text so the model can condition its responses on the type of mistake being made.

Dataset Preparation

The raw dataset is stored in dataset.jsonl.
A preprocessing script (prepare_data.py) converts each example into a structured text format that includes:

The problem statement

The incorrect student answer

The error type

An explanation of the mistake

A correction strategy

The correct explanation

This structured format is written to formatted_dataset.jsonl, which is used for training.

Model and Training Setup

The base model used is gpt2-medium, chosen to balance model capacity and CPU feasibility.

Fine-tuning is performed using LoRA (Low-Rank Adaptation):

Only attention layers are adapted

Token embeddings are frozen

This keeps training lightweight and efficient

The model is trained using a standard causal language modeling objective, meaning it learns to generate the structured correction text token by token.

The purpose of training is to bias behavior, not to enforce correctness.

Inference and Decoding

During inference, the model is prompted with:

A physics problem

A student’s incorrect answer

A task instruction requesting a correction

Different decoding strategies were tested. The project intentionally explores how:

Greedy decoding can cause early termination

Sampling can cause topic drift or repetition

Anti-repetition constraints can lead to structural noise

These behaviors are documented as part of the analysis.

Observed Behavior

After fine-tuning, the model often:

Recognizes that an answer is incorrect

Attempts correction-style explanations

Produces structured responses similar to training data

However, the model also exhibits:

Repetition loops

Conceptual hallucinations

Incorrect physics explanations

Sensitivity to prompt phrasing and decoding parameters

These behaviors are expected given the small dataset, small base model, and absence of instruction tuning or reinforcement learning.

Key Insight

Error-aware fine-tuning biases model behavior but does not guarantee correctness.

This project demonstrates why:

Instruction-tuned models

Larger datasets

Reward modeling and RLHF

are necessary for reliable reasoning systems.

The limitations observed here are a feature of the study, not a flaw.

Why This Project Matters

This project demonstrates:

Thoughtful dataset design

Explicit modeling of reasoning errors

Parameter-efficient fine-tuning

Analysis of LLM failure modes

Practical limitations of small causal language models

Rather than hiding failures, the project analyzes and explains them, which is essential for real research work.

Limitations

This project does not aim to:

Guarantee correct physics answers

Replace symbolic reasoning

Compete with instruction-tuned models

All results should be interpreted as behavioral tendencies, not correctness guarantees.

Future Work

Possible extensions include:

Training on instruction-tuned base models

Adding reward models for correction quality

Scaling the dataset

Automatic error-type classification

Human evaluation of explanations

Disclaimer

This project is for educational and research purposes only.
It is not intended for deployment or real-world tutoring systems.