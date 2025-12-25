---

# Error-Aware Fine-Tuning for Scientific Reasoning

## Overview

This project explores **error-aware fine-tuning of large language models (LLMs)** for physics and scientific reasoning.

Instead of training a model only on correct solutions, the system is trained on **incorrect student answers paired with explanations of why they are wrong and how to correct them**.

The goal is **not** to build a perfect physics solver.
Instead, the project studies how **parameter-efficient fine-tuning (LoRA)** biases a language model’s behavior toward recognizing mistakes, attempting corrective explanations, and where such approaches **fail**.

The emphasis is on **model behavior analysis and limitations**, rather than raw accuracy.

---

## Core Idea

Most fine-tuning pipelines train models only on correct answers.
This project explicitly models **student errors**.

Each training example contains:

* A physics problem
* A student’s **incorrect** answer
* The **type of error** made
* An explanation of **why** the answer is wrong
* A **correction strategy**
* The correct explanation and final answer

By exposing the model to these patterns, the project investigates whether the model learns to:

* Detect that an answer is incorrect
* Attempt a correction
* Produce structured explanatory responses

---

## Error Taxonomy

The project defines a small but meaningful taxonomy of error types, including:

* Conceptual errors
* Formula misuse
* Calculation errors
* Unit errors
* Sign errors
* Assumption errors

These error labels are embedded **directly in the training text**, allowing the model to condition its responses on the type of mistake being made.

---

## Dataset Preparation

* Raw data is stored in `dataset.jsonl`
* Each entry represents a **single incorrect student response**

A preprocessing script (`prepare_data.py`) converts each example into a structured training format containing:

* Problem statement
* Incorrect student answer
* Error type
* Explanation of the mistake
* Correction strategy
* Correct explanation

The processed output is written to `formatted_dataset.jsonl`, which is used for training.

---

## Model and Training Setup

* **Base model:** `gpt2-medium`

  * Chosen to balance model capacity and CPU feasibility

* **Fine-tuning method:** LoRA (Low-Rank Adaptation)

  * Only attention layers are adapted
  * Token embeddings are frozen
  * Training remains lightweight and efficient

* **Training objective:**
  Standard causal language modeling
  (the model learns to generate correction-style text token by token)

The purpose of training is to **bias behavior**, not to enforce correctness.

---

## Inference and Decoding

During inference, the model is prompted with:

* A physics problem
* A student’s incorrect answer
* A task instruction requesting a correction

Different decoding strategies were intentionally explored:

* Greedy decoding → early termination
* Sampling → topic drift and repetition
* Anti-repetition penalties → structural noise

These behaviors are **documented as part of the analysis**, not hidden.

---

## Observed Behavior

After fine-tuning, the model often:

* Recognizes that an answer is incorrect
* Attempts correction-style explanations
* Produces structured responses similar to the training data

However, the model also exhibits:

* Repetition loops
* Conceptual hallucinations
* Incorrect physics explanations
* High sensitivity to prompt phrasing and decoding parameters

These behaviors are expected given:

* Small dataset size
* Small base model
* No instruction tuning
* No reinforcement learning

---

## Key Insight

**Error-aware fine-tuning biases model behavior but does not guarantee correctness.**

This project demonstrates why:

* Instruction-tuned models
* Larger datasets
* Reward modeling
* RLHF-style optimization

are necessary for reliable scientific reasoning systems.

The limitations observed here are a **feature of the study, not a flaw**.

---

## Why This Project Matters

This project demonstrates:

* Thoughtful dataset design
* Explicit modeling of reasoning errors
* Parameter-efficient fine-tuning
* Analysis of LLM failure modes
* Practical limits of small causal language models

Rather than hiding failures, the project **analyzes and explains them**, which is essential for real research work.

---

## Limitations

This project does **not** aim to:

* Guarantee correct physics answers
* Replace symbolic or rule-based reasoning
* Compete with instruction-tuned foundation models

All results should be interpreted as **behavioral tendencies**, not correctness guarantees.

---

## Future Work

Possible extensions include:

* Training on instruction-tuned base models
* Adding reward models for correction quality
* Scaling the dataset significantly
* Automatic error-type classification
* Human evaluation of explanations

---

## Disclaimer

This project is for **educational and research purposes only**.
It is **not intended for deployment** or real-world tutoring systems.

---
