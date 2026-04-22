# First-Principles Constrained Decoding

This project serves as an educational sandbox to learn exactly how **constrained decoding** and **structured generation** work under the hood. 

Instead of treating the Large Language Model (LLM) as a black box where we use prompt engineering to beg for JSON—or using a high-level library like `outlines` that handles the magic for you—we will manually intercept the model's generation loop to mathematically guarantee structured output.

## The Problem: Why not Post-Hoc Parsing?

When extracting data from a document, asking an LLM to output JSON and then running `json.loads(output)` is brittle. The model might hallucinate extra fields, drop quotes, generate unescaped characters, or add conversational filler like *"Here is your JSON:"*. 

Post-hoc parsing is a reactive trial-and-error method. If the LLM generates bad output, your pipeline breaks.

## The Pedagogy: Manipulating Logits directly

At its core, an autoregressive LLM is a classification model. At every step $t$, it looks at the context and outputs a probability distribution (logits) across its entire vocabulary (e.g., ~32,000 to ~128,000 possible next tokens).

**Constrained decoding intercepts these raw logits right before the model picks the next token.** 

By building a Finite State Machine (FSM) that understands our target JSON syntax, we can forbid the model from predicting invalid tokens by setting their logit scores to $-\infty$. The model becomes physically incapable of generating malformed JSON.

---

## Implementation Plan

To deeply understand structured generation, we will build the constraint engine ourselves using HuggingFace `transformers`. We will extract text from a PDF, and run the pipeline locally.

### Step 1: Environment Setup
```bash
pip install pymupdf transformers torch
```

### Step 2: PDF Extractor (`extractor.py`)
A fast utility leveraging `PyMuPDF` to load a `.pdf` file and extract its raw text content into a string. 

### Step 3: The Finite State Machine (`fsm.py`)
To enforce structure, the pipeline needs to know where it is in the generation process. We will code a simple, hardcoded deterministic FSM for a basic schema (e.g., `{"name": "<STRING>", "age": <NUMBER>}`). 

Our states will look like this:
*   `STATE_0`: System is waiting to generate exactly `{"name": "`
*   `STATE_1`: System is generating string characters (waiting for a closing `"`)
*   `STATE_2`: System is waiting to generate exactly `, "age": `
*   `STATE_3`: System is generating sequential digits `[0-9]+`
*   `STATE_4`: System is waiting to generate exactly `}` and then `<EOS>`

### Step 4: The Custom LogitsProcessor (`processor.py`)
This is the heart of the project. We will subclass HuggingFace's `LogitsProcessor`. 
1.  At every generation step, this class receives the `input_ids` generated so far and the raw `scores` (logits) predicted for the next token.
2.  It decodes the recent tokens to check the `FSM` state and see what characters are legally allowed next.
3.  It runs a filter over the tokenizer's vocabulary. If a vocabulary token starts with an illegal character for our current state, we mask it out: `scores[:, token_id] = -float('inf')`.

*Note: This step forces you to confront the tricky reality of sub-word tokenization—a single word might span three tokens, and your constraints must handle partial words!*

### Step 5: End-to-End Pipeline (`main.py`)
We wire it all together: 
1. Parse the PDF.
2. Load a small local LLM (e.g., Llama-3.2-1B).
3. Initialize our `FSMLogitsProcessor` and attach it via the `logits_processor` argument in `model.generate(...)`.
4. Watch the model perfectly output our schema.
