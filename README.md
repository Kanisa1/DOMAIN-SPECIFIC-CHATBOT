
# 🩺 MediBot Africa – Domain-Specific Healthcare Chatbot

**Owner:** Kanisa Thiak
**Model:** `microsoft/DialoGPT-medium`
**Framework:** Hugging Face Transformers + PyTorch

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Collection and Preprocessing](#2-dataset-collection-and-preprocessing)
3. [Model Selection and Fine-Tuning](#3-model-selection-and-fine-tuning)
4. [Performance Metrics and Evaluation](#4-performance-metrics-and-evaluation)
5. [User Interface Integration](#5-user-interface-integration)
6. [Code Quality and Documentation](#6-code-quality-and-documentation)
7. [Challenges and Solutions](#7-challenges-and-solutions)
8. [Project Setup and Navigation](#8-project-setup-and-navigation)
9. [Demo and Submission Artifacts](#9-demo-and-submission-artifacts)
10. [Conclusion and Future Work](#10-conclusion-and-future-work)
11. [References](#11-references)

---

## 1. 🧠 Project Overview

**MediBot Africa** is a **domain-specific healthcare chatbot** developed to provide **accurate, accessible, and contextually relevant medical information** to users across Africa.

Built on **Transformer-based models**, the chatbot delivers **safe, coherent responses** regarding common illnesses, symptoms, treatments, and preventive measures.

### Domain Relevance

Healthcare was selected for its **social and humanitarian impact**. Challenges in African communities include:

* Widespread misinformation about medical conditions
* Limited healthcare infrastructure
* Low digital health literacy

MediBot Africa contributes to **UN SDG 3 – Good Health and Well-being**, providing **reliable digital medical guidance**.

---

## 2. 🩸 Dataset Collection and Preprocessing

### Dataset Description

A **custom conversational dataset** (`medibot_dataset.json`) was curated to reflect **realistic African healthcare scenarios**, including dialogues about:

* Malaria
* Typhoid
* Fever
* Pneumonia
* Vaccination awareness

### Preprocessing Steps

1. **Text normalization:** Lowercasing, punctuation cleanup
2. **Noise removal:** Excluding incomplete or irrelevant dialogues
3. **Handling missing values:** Dropping null or empty entries
4. **Tokenization:** Using `AutoTokenizer` (max length = 512)
5. **Formatting:** Structured into JSON and loaded via Pandas

**Result:** A **clean, balanced dataset** optimized for Transformer fine-tuning.

---

## 3. ⚙️ Model Selection and Fine-Tuning

### Model Iteration

| Model               | Observation                                   | Outcome     |
| ------------------- | --------------------------------------------- | ----------- |
| **T5-Small**        | Efficient but hallucinated medical facts      | ❌ Rejected  |
| **T5-Base**         | Improved fluency, still prone to errors       | ⚠️ Moderate |
| **DialoGPT-Medium** | Contextually rich, stable dialogue generation | ✅ Selected  |

### Fine-Tuning Configuration

| Hyperparameter      | Value                       |
| ------------------- | --------------------------- |
| Model               | `microsoft/DialoGPT-medium` |
| Learning Rate       | 5e-5                        |
| Batch Size          | 4                           |
| Epochs              | 3                           |
| Max Sequence Length | 512                         |

**Training Highlights:** GPU acceleration, dynamic padding, attention masking, checkpoint saving.

### Key Observations

* Reduced hallucinations compared to T5 models
* Improved context retention and dialogue coherence
* Validation loss steadily decreased, confirming stable training

---

## 4. 📊 Performance Metrics and Evaluation

### Quantitative Metrics

| Metric     | Result | Interpretation                         |
| ---------- | ------ | -------------------------------------- |
| BLEU Score | 0.1706 | Moderate alignment with reference text |
| ROUGE-1    | 0.3301 | Good lexical overlap                   |
| ROUGE-2    | 0.3205 | Strong phrase-level relevance          |
| ROUGE-L    | 0.3337 | Structural similarity                  |
| F1-Score   | 0.62   | Balanced precision and recall          |
| Perplexity | 18.7   | Low uncertainty, fluent generation     |

### Qualitative Evaluation

* Accurate responses to domain-specific medical queries
* Rejection of unsafe or unrelated questions
* Natural, human-like conversational flow

---

## 5. 💬 User Interface Integration

### Interface Design

* **Gradio-based web app** for real-time interaction
* **Responsive layout** for desktop and mobile
* Text input for queries and dynamic chatbot responses
* Deployed on **Hugging Face Spaces** for public access

---

## 6. 🧩 Code Quality and Documentation

* **Modular structure:** Separate sections for configuration, preprocessing, training, and evaluation
* **Readable code:** Clear naming conventions and detailed comments
* **Version control:** Git + Hugging Face synchronization
* **Reproducibility:** All parameters, datasets, and checkpoints documented

---

## 7. 🧠 Challenges and Solutions

| Challenge                   | Description                           | Solution                                             |
| --------------------------- | ------------------------------------- | ---------------------------------------------------- |
| Model hallucination         | T5 models generated off-topic answers | Switched to DialoGPT with optimized learning rate    |
| Hugging Face hosting limits | Model size >1GB                       | Split model and interface into separate repositories |
| GitHub push limits          | Checkpoints exceeded file size cap    | Linked external model repo                           |
| W&B integration             | Dependency conflicts                  | Deferred integration post-stabilization              |
| Perplexity computation      | GPU memory overflow                   | Adopted batch evaluation + gradient checkpointing    |

---

## 8. 🛠️ Project Setup and Navigation

### Prerequisites

* Python 3.9+
* PyTorch 2.x
* Hugging Face Transformers 5.x
* Gradio 3.x

### Installation

```bash
# Clone repository
git clone https://github.com/Kanisa12/medibot-africa.git
cd medibot-africa

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Chatbot

```bash
# Launch Gradio interface
python chat.py
```

### Project Structure

```
medibot-africa/
├── data/                  # Dataset files
│   └── medibot_dataset.json
├── notebooks/             # Jupyter notebooks for experimentation
│   └── chat.ipynb
├── models/                # Trained model checkpoints
├── scripts/               # Preprocessing and training scripts
├── chat.py                # Entry point for Gradio UI
├── requirements.txt       # Python dependencies
└── README.md
```

---

## 9. 🎥 Demo and Submission Artifacts

* **Demo Video:** [YouTube Link](https://youtu.be/XGkV0HVm7xs?si=AE9sR9_YTDEG7Z0g)
* **Model Repository:** [Hugging Face – Kanisa12/medibot-africa](https://huggingface.co/Kanisa12/medibot-africa)
* **Interface Repository:** [Hugging Face – MediBot Interface](#)

---

## 10. 🏁 Conclusion and Future Work

**MediBot Africa** demonstrates that **Transformer-based dialogue models** can improve healthcare education and awareness in Africa.

### Future Enhancements

* Expand dataset with **multilingual support** (English, Swahili, Arabic)
* Integrate **real-time F1-score tracking** and **Weights & Biases logging**
* Deploy as a **RESTful API** for mobile apps
* Improve **model explainability** for transparent medical insights

---

## 11. 🔗 References

1. Hugging Face, *Transformers Documentation*, 2024 — [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
2. Paszke, A. et al., *PyTorch: An Imperative Style, High-Performance Deep Learning Library*, NeurIPS 2019 — [https://pytorch.org](https://pytorch.org)
3. TensorFlow Team, *TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems*, 2015 — [https://tensorflow.org](https://tensorflow.org)
4. Papineni, K. et al., *BLEU: A Method for Automatic Evaluation of Machine Translation*, ACL 2002 — [https://aclanthology.org/P02-1040](https://aclanthology.org/P02-1040)
5. Lin, C.-Y., *ROUGE: A Package for Automatic Evaluation of Summaries*, ACL Workshop 2004 — [https://aclanthology.org/W04-1013](https://aclanthology.org/W04-1013)
6. Microsoft, *DialoGPT: Large-Scale Generative Pre-training for Conversational Response Generation*, 2020 — [https://huggingface.co/microsoft/DialoGPT-medium](https://huggingface.co/microsoft/DialoGPT-medium)
7. Raffel, C. et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)*, JMLR, 2020

---

### 🩷 *“MediBot Africa – Empowering Health Through AI Conversations.”*

---
