# Urdu Question Generation using Seq2Seq

An end-to-end **Urdu Question Generation (QG)** system that generates natural-language questions from Urdu context passages and marked answer spans.

This project implements a custom **BiLSTM Encoder–Decoder architecture with Bahdanau Attention**, trained on the UQA dataset and deployed as an interactive web application using **Hugging Face Spaces, Gradio, and ZeroGPU**.

---

## Live Demo

### Hugging Face Space

[Open Live Demo](https://huggingface.co/spaces/razaasherazi/urdu-qgen-demo)

The deployed application provides:

- Urdu context input
- Answer-span based question generation
- Greedy decoding
- Beam Search decoding
- Adjustable beam width
- Interactive Gradio interface
- GPU-accelerated inference through Hugging Face ZeroGPU

### Hugging Face Model

[View Trained Model](https://huggingface.co/razaasherazi/urdu-qgen-seq2seq)

---

## Project Overview

Question Generation is the task of automatically generating a question from a given passage and its corresponding answer.

For example:

### Input

```text
پاکستان کا دارالحکومت <ans> اسلام آباد </ans> ہے۔
```

### Generated Question

```text
پاکستان کا دارالحکومت کیا ہے؟
```

The `<ans>` and `</ans>` markers identify the answer span that the generated question should target.

The overall system follows this pipeline:

```text
UQA Dataset
     |
     v
Data Preparation
     |
     v
Answer Span Marking
     |
     v
SentencePiece Tokenization
     |
     v
BiLSTM Encoder
     |
     v
Bahdanau Attention
     |
     v
LSTM Decoder
     |
     v
Greedy / Beam Search Decoding
     |
     v
Generated Urdu Question
     |
     v
Gradio Web Application
     |
     v
Hugging Face ZeroGPU
```

---

## Features

* Urdu Question Generation
* Custom Seq2Seq neural architecture
* Bidirectional LSTM Encoder
* Multi-layer LSTM Decoder
* Bahdanau Attention mechanism
* SentencePiece subword tokenization
* Answer-span markers
* Teacher forcing during training
* Greedy decoding
* Beam Search decoding
* Configurable beam width
* Validation perplexity tracking
* Token-level accuracy evaluation
* UQA validation evaluation
* Wiki-UQA out-of-domain evaluation
* Attention visualization
* Interactive Gradio frontend
* Hugging Face model hosting
* Hugging Face ZeroGPU deployment

---

# Model Architecture

The model is a neural sequence-to-sequence architecture consisting of four main components.

## 1. Encoder

A **2-layer Bidirectional LSTM** encodes the Urdu input sequence.

Configuration:

| Parameter           | Value |
| ------------------- | ----: |
| Embedding Dimension |   256 |
| Hidden Dimension    |   512 |
| Encoder Layers      |     2 |
| Bidirectional       |   Yes |
| Dropout             |   0.3 |

The bidirectional encoder processes the input from both directions and captures contextual information from the complete passage.

---

## 2. Bahdanau Attention

Bahdanau Attention allows the decoder to dynamically focus on different parts of the encoded input while generating each question token.

This is particularly useful for Question Generation because the model needs to identify the relevant information surrounding the marked answer span.

At each decoding step, the attention mechanism calculates relevance scores between the current decoder state and the encoder outputs.

The resulting context vector is then provided to the decoder to help generate the next question token.

---

## 3. Decoder

The decoder is a **2-layer LSTM**.

At every decoding step, it receives:

* Previous generated token
* Decoder hidden state
* Decoder cell state
* Encoder outputs
* Attention context

The decoder then predicts the next token in the question.

---

## 4. Output Layer

The decoder hidden representation is passed through a linear projection layer to produce a probability distribution over the SentencePiece vocabulary.

The vocabulary contains:

```text
8,000 tokens
```

---

# Special Tokens

The tokenizer uses the following special tokens:

| Token    | ID | Purpose               |
| -------- | -: | --------------------- |
| `<pad>`  |  0 | Padding               |
| `<unk>`  |  1 | Unknown token         |
| `<s>`    |  2 | Beginning of sequence |
| `</s>`   |  3 | End of sequence       |
| `<ans>`  |  - | Answer span start     |
| `</ans>` |  - | Answer span end       |

The answer markers are inserted into the source context so that the model knows which information the generated question should target.

---

# Dataset

The main training dataset is **UQA**, containing Urdu question-answer examples.

After preprocessing:

| Split      | Examples |
| ---------- | -------: |
| Training   |   75,069 |
| Validation |   10,014 |

An additional **Wiki-UQA** dataset is used for out-of-domain evaluation.

Processed Wiki-UQA:

```text
178 examples
```

---

# Data Preparation

The preprocessing pipeline performs the following operations:

1. Load the raw UQA dataset.
2. Extract context, question, and answer information.
3. Identify answer spans.
4. Insert `<ans>` and `</ans>` markers into the context.
5. Create source-target pairs.
6. Remove invalid examples.
7. Save the processed data as TSV files.

Generated files:

```text
data/processed/
├── train.tsv
├── valid.tsv
└── wiki_valid.tsv
```

Each example follows the format:

```text
source<TAB>target
```

Example:

```text
پاکستان کا دارالحکومت <ans> اسلام آباد </ans> ہے۔    پاکستان کا دارالحکومت کیا ہے؟
```

---

# Tokenization

The project uses **SentencePiece** for Urdu subword tokenization.

Configuration:

| Parameter          |             Value |
| ------------------ | ----------------: |
| Vocabulary Size    |             8,000 |
| Model Type         |           Unigram |
| Character Coverage |               1.0 |
| Answer Tokens      | `<ans>`, `</ans>` |

Tokenizer files:

```text
tokenizer/
├── sp_corpus.txt
├── ur_sp.model
└── ur_sp.vocab
```

SentencePiece provides subword-level tokenization, which is useful for Urdu because it can represent rare and previously unseen word forms more effectively than a purely word-level tokenizer.

---

# Training

The model was trained using **PyTorch**.

Training objective:

```text
Cross Entropy Loss
```

Padding tokens are ignored during loss calculation.

Optimizer:

```text
Adam
```

The training process uses teacher forcing to help the decoder learn the target question sequence.

Gradient clipping was also used in Experiment 2 to improve training stability.

---

# Experiments

Two training configurations were evaluated.

## Experiment 1

| Parameter             | Value |
| --------------------- | ----: |
| Learning Rate         |  1e-3 |
| Teacher Forcing Ratio |   0.5 |
| Epochs Completed      |     4 |
| Best Epoch            |     2 |

Best validation result:

```text
Validation Loss: 5.9067
Validation Perplexity: 367.51
```

Teacher-forced token accuracy:

```text
Training Accuracy: 38.34%
Validation Accuracy: 31.45%
```

Experiment 1 achieved the strongest validation loss among the completed experiments.

---

## Experiment 2

Experiment 2 was trained with a lower learning rate, stronger teacher forcing, gradient clipping, and a batch size of 64.

| Parameter             | Value |
| --------------------- | ----: |
| Learning Rate         |  3e-4 |
| Teacher Forcing Ratio |   0.8 |
| Batch Size            |    64 |
| Gradient Clipping     |   1.0 |
| Epochs Completed      |     6 |
| Best Epoch            |     1 |

Results:

```text
Best Validation Loss:       6.6392
Best Validation Perplexity: 764.47

Training Token Accuracy:    31.79%
Validation Token Accuracy:  29.58%
```

The Experiment 2 checkpoint was retained as the final deployed model.

---

# Final Model

The deployed model is:

```text
experiment-2/best_model.pt
```

Model configuration:

| Parameter             |      Value |
| --------------------- | ---------: |
| Embedding Dimension   |        256 |
| Hidden Dimension      |        512 |
| Encoder Layers        |          2 |
| Decoder Layers        |          2 |
| Bidirectional Encoder |        Yes |
| Dropout               |        0.3 |
| Vocabulary Size       |      8,000 |
| Maximum Target Length |         30 |
| Trainable Parameters  | 35,505,984 |

The trained model and associated files are hosted on Hugging Face.

[Hugging Face Model Repository](https://huggingface.co/razaasherazi/urdu-qgen-seq2seq)

---

# Decoding

The system supports two decoding strategies.

## Greedy Decoding

Greedy decoding selects the highest-probability token at every decoding step.

Conceptually:

```text
At each step:
    select the token with the highest probability
```

Advantages:

* Fast
* Simple
* Low computational cost

---

## Beam Search

Beam Search maintains multiple candidate sequences during generation.

The frontend allows the user to control the beam width.

For example:

```text
Beam Width = 5
```

means that the decoder keeps the five strongest candidate sequences during generation.

Beam Search provides a broader search over possible questions compared with greedy decoding.

---

# Evaluation

The project includes multiple evaluation and analysis procedures.

Evaluation includes:

* Validation Loss
* Perplexity
* Teacher-forced Token Accuracy
* Generated Question Examples
* Failure-case Analysis
* Wiki-UQA Out-of-Domain Evaluation
* Greedy Decoding
* Beam Search Decoding
* Attention Visualization

The evaluation pipeline generates questions from validation examples and compares model behavior against reference questions.

---

# Dataset Diagnostics

The processed dataset was also analyzed to verify data quality and sequence lengths.

Answer markers were successfully identified in:

```text
Training examples:   75,069 / 75,069
Validation examples: 10,014 / 10,014
```

Approximate source token statistics:

| Split      |  Mean | Median | P95 | Maximum |
| ---------- | ----: | -----: | --: | ------: |
| Training   | 43.86 |     42 |  71 |     197 |
| Validation | 44.97 |     43 |  72 |     148 |

Approximate target token statistics:

| Split      |  Mean | Median | P95 | Maximum |
| ---------- | ----: | -----: | --: | ------: |
| Training   | 17.15 |     16 |  27 |      56 |
| Validation | 17.92 |     17 |  28 |      47 |

Only a small portion of examples exceeded the configured maximum target length of 30 tokens.

This indicates that the dataset preprocessing and sequence-length distribution were generally reasonable.

---

# Example

## Input Context

```text
نارمن (Norman: Nourmands؛ French: Normands؛ Latin: Normanni)
وہ لوگ تھے جنہوں نے 10 ویں اور 11 ویں صدیوں میں
<ans> فرانس </ans> کے ایک خطے نارمنڈی کو اپنا نام دیا۔
```

## Reference Question

```text
نارمنڈی کس ملک میں واقع ہے؟
```

The current model can generate Urdu question-like outputs, but generation quality is not yet consistently comparable to the reference questions.

This reflects the current experimental state of the project and provides opportunities for further model improvement.

---

# Frontend

The model is deployed through an interactive **Gradio** web application.

The frontend provides:

```text
Urdu Context
     |
     v
Beam Width Selection
     |
     v
Generate Question
     |
     +----------------------+
     |                      |
     v                      v
Greedy Decoding       Beam Search
     |                      |
     v                      v
Generated Question    Generated Question
```

Frontend files:

```text
frontend/
├── app.py
├── requirements.txt
├── ur_sp.model
└── ur_sp.vocab
```

---

# Deployment

The application is deployed using:

```text
Hugging Face Spaces
        +
Gradio
        +
ZeroGPU
```

The deployed application downloads the trained model and tokenizer directly from the Hugging Face model repository.

Model repository:

```text
razaasherazi/urdu-qgen-seq2seq
```

Application repository:

```text
razaasherazi/urdu-qgen-demo
```

### ZeroGPU

The application uses Hugging Face ZeroGPU for GPU-accelerated inference.

ZeroGPU dynamically allocates GPU resources when inference is requested, allowing the application to run without maintaining a dedicated GPU server continuously.

---

# Live Application

The final application is publicly deployed at:

**Hugging Face Spaces:**

[https://huggingface.co/spaces/razaasherazi/urdu-qgen-demo](https://huggingface.co/spaces/razaasherazi/urdu-qgen-demo)

Users can provide an Urdu context containing an answer span and generate a question using:

* Greedy Decoding
* Beam Search
* Configurable Beam Width

---

# Repository Structure

```text
urdu-qgen-seq2seq/
│
├── configs/
│   └── config.py
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── train.tsv
│       ├── valid.tsv
│       └── wiki_valid.tsv
│
├── tokenizer/
│   ├── sp_corpus.txt
│   ├── ur_sp.model
│   └── ur_sp.vocab
│
├── src/
│   ├── data/
│   │   └── prepare_data.py
│   │
│   ├── model/
│   │   ├── attention.py
│   │   ├── decoder.py
│   │   ├── encoder.py
│   │   └── seq2seq.py
│   │
│   ├── training/
│   │   └── ...
│   │
│   ├── decoding/
│   │   └── ...
│   │
│   ├── evaluation/
│   │   └── ...
│   │
│   └── tokenizer/
│       └── ...
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_tokenizer_and_model.ipynb
│   ├── 03_full_training.ipynb
│   ├── 04_evaluation.ipynb
│   ├── 05_frontend_app.ipynb
│   └── 06_complete_pipeline_kaggle.ipynb
│
├── frontend/
│   ├── app.py
│   ├── requirements.txt
│   ├── ur_sp.model
│   └── ur_sp.vocab
│
├── checkpoints/
│
├── results/
│
├── docs/
│
├── tests/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Notebooks

The project development and experimentation are organized into notebooks.

## `01_data_preparation.ipynb`

Responsible for:

* Loading UQA data
* Processing answer spans
* Creating training and validation TSV files
* Preparing Wiki-UQA evaluation data
* Inspecting processed examples

---

## `02_tokenizer_and_model.ipynb`

Responsible for:

* Training the SentencePiece tokenizer
* Defining the model architecture
* Testing tokenization
* Testing model components

---

## `03_full_training.ipynb`

Responsible for:

* Training the Seq2Seq model
* Tracking training loss
* Tracking validation loss
* Calculating validation perplexity
* Saving model checkpoints

---

## `04_evaluation.ipynb`

Responsible for:

* Loading trained checkpoints
* Performing question generation
* Evaluating validation data
* Performing Wiki-UQA evaluation
* Comparing decoding strategies
* Analyzing failure cases
* Visualizing attention
* Saving evaluation results

---

## `05_frontend_app.ipynb`

Responsible for:

* Preparing the deployment frontend
* Testing the inference pipeline
* Integrating the trained model with the frontend
* Preparing deployment files

---

## `06_complete_pipeline_kaggle.ipynb`

Provides a consolidated version of the complete Kaggle workflow, including the major stages of the project pipeline.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/RazaSherazi09/urdu-qgen-seq2seq.git
cd urdu-qgen-seq2seq
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

### Windows

```bash
venv\Scripts\activate
```

### Linux / WSL

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Frontend Locally

Install the frontend dependencies:

```bash
pip install -r frontend/requirements.txt
```

Run the frontend:

```bash
streamlit run frontend/app.py
```

The local application will then be available at the Streamlit URL displayed in the terminal.

---

# Reproducing the Pipeline

The recommended workflow is:

```text
1. Prepare Dataset
        |
        v
2. Train SentencePiece Tokenizer
        |
        v
3. Build Model
        |
        v
4. Train Model
        |
        v
5. Evaluate Model
        |
        v
6. Save Best Checkpoint
        |
        v
7. Upload Model to Hugging Face
        |
        v
8. Deploy Gradio Application
```

---

# Technologies Used

| Technology          | Purpose                   |
| ------------------- | -------------------------- |
| Python              | Programming Language      |
| PyTorch             | Deep Learning Framework   |
| NumPy               | Numerical Computing       |
| Pandas              | Data Processing           |
| SentencePiece       | Subword Tokenization      |
| Streamlit           | Local Frontend            |
| Gradio              | Web Application Interface |
| Hugging Face Hub    | Model Hosting             |
| Hugging Face Spaces | Application Deployment    |
| ZeroGPU             | GPU-Accelerated Inference |
| Kaggle              | Training Environment      |
| GitHub              | Source Code Management    |

---

# Limitations

The current model demonstrates a complete Urdu Question Generation pipeline, but the generated question quality is not yet consistently high.

Observed limitations include:

* Repetitive generated questions
* Incorrect question structures
* Occasionally irrelevant generated questions
* Weak generalization on some validation examples
* Beam Search does not always improve semantic quality
* Limited training time restricted the number of completed epochs
* The current model is relatively large for CPU-only inference
* Further hyperparameter tuning is required

The project should therefore be considered a functional **research prototype** rather than a production-level Urdu Question Generation system.

---

# Future Improvements

## Model Improvements

* Increase training duration
* Tune learning rate
* Tune teacher-forcing ratio
* Experiment with different embedding sizes
* Experiment with larger hidden dimensions
* Experiment with Transformer-based architectures
* Use pretrained multilingual or Urdu language models
* Improve answer-span representations

## Data Improvements

* Increase Urdu training data
* Improve dataset cleaning
* Remove noisy question-answer pairs
* Add more diverse Urdu sources
* Improve out-of-domain evaluation

## Decoding Improvements

* Tune Beam Search parameters
* Add length normalization
* Add repetition penalties
* Compare nucleus sampling
* Compare top-k sampling
* Implement diverse Beam Search

## Evaluation Improvements

* Add BLEU
* Add ROUGE
* Add semantic similarity metrics
* Add BERTScore or multilingual semantic metrics
* Perform human evaluation of question quality

## Deployment Improvements

* Optimize model size
* Add model quantization
* Improve inference latency
* Add caching
* Improve frontend validation
* Add more example contexts
* Add downloadable generated results

---

# Project Status

| Component          | Status     |
| ------------------ | ---------- |
| Data Preparation   | Completed  |
| Tokenizer          | Completed  |
| Model Architecture | Completed  |
| Training           | Completed  |
| Evaluation         | Completed  |
| Hugging Face Model | Uploaded   |
| Frontend           | Completed  |
| Hugging Face Space | Deployed   |
| ZeroGPU            | Configured |
| Public Demo        | Ready      |

---

# Conclusion

This project implements a complete **Urdu Question Generation system** using a custom neural Seq2Seq architecture with a Bidirectional LSTM Encoder, Bahdanau Attention, and LSTM Decoder.

The system covers the complete machine-learning lifecycle:

```text
Dataset
   |
   v
Preprocessing
   |
   v
Tokenization
   |
   v
Model Design
   |
   v
Training
   |
   v
Evaluation
   |
   v
Model Hosting
   |
   v
Web Deployment
```

The final system is available through an interactive Hugging Face Space, allowing users to provide Urdu context and generate questions using both Greedy Decoding and Beam Search.

The project provides a foundation for future research into higher-quality Urdu Question Generation using larger datasets, improved training strategies, pretrained language models, and more advanced decoding techniques.

---

# Author

**Raza Sherazi**

AI Engineer / Machine Learning Developer

GitHub: [RazaSherazi09](https://github.com/RazaSherazi09)

Hugging Face: [razaasherazi](https://huggingface.co/razaasherazi)

---

## Links

* **GitHub Repository:** [https://github.com/RazaSherazi09/urdu-qgen-seq2seq](https://github.com/RazaSherazi09/urdu-qgen-seq2seq)
* **Hugging Face Model:** [https://huggingface.co/razaasherazi/urdu-qgen-seq2seq](https://huggingface.co/razaasherazi/urdu-qgen-seq2seq)
* **Live Demo:** [https://huggingface.co/spaces/razaasherazi/urdu-qgen-demo](https://huggingface.co/spaces/razaasherazi/urdu-qgen-demo)
