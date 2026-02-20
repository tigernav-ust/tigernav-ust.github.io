<h1 align="center">🐯 TigerNav</h1>

<p align="center">
  <em>TigerNav: Development of a Virtual Assistant Using an Autoregressive Model for Indoor Navigation</em>
</p>

<p align="center">
  <img alt="TENCON 2025 Accepted" src="https://img.shields.io/badge/TENCON%202025-Accepted-2ea44f?style=flat">
  <img alt="DOI 10.1109/TENCON66050.2025.11374916" src="https://img.shields.io/badge/DOI-10.1109%2FTENCON66050.2025.11374916-2563eb?style=flat">
  <img alt="License MIT" src="https://img.shields.io/badge/License-MIT-f59e0b?style=flat">
</p>

<p align="center">
  📄 <a href="https://doi.org/10.1109/TENCON66050.2025.11374916"><strong>Paper</strong></a>
  &nbsp;•&nbsp;
  🌐 <a href="https://tigernav-ust.github.io/"><strong>Project Page</strong></a>
  &nbsp;•&nbsp;
  💻 <a href="https://github.com/tigernav-ust/tigernav-ust.github.io"><strong>Code</strong></a>
</p>


---

## 📄 Abstract

TigerNav is an intelligent campus navigation chatbot developed to assist students, faculty, and visitors in navigating university grounds through conversational interaction. The system integrates instruction-tuned Large Language Models (LLMs) fine-tuned on campus-specific navigation data.  

We evaluate multiple training paradigms including Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and Odds Ratio Preference Optimization (ORPO) to improve contextual accuracy and response alignment. Results demonstrate that preference-based optimization enhances semantic alignment and reduces perplexity compared to baseline fine-tuning methods.

---

## 🗺️ System Overview

TigerNav follows a structured conversational pipeline:

1. **User Query Input**
2. **Intent & Context Processing**
3. **Fine-Tuned LLM Inference**
4. **Response Generation**

The model was trained on structured campus navigation data and optimized using preference learning techniques to improve real-world conversational performance.

---

## 📊 Results

Performance evaluation metrics include:

- **METEOR** – Semantic alignment  
- **BERTScore** – Contextual similarity  
- **Perplexity** – Model confidence  

Preference optimization (ORPO) showed improved alignment and lower perplexity relative to standard SFT.

For complete experimental results, please refer to the published paper.

---

## 📁 Repository Structure

```
tigernav-ust.github.io/
│
├── docs/                     # GitHub Pages site
├── codes/                    # Sanitized training & evaluation scripts
│   ├── finetuning_trainer.py
│   ├── finetuning_orpo.py
│   ├── DPO_Format.py
│   ├── DPO_dataset_metric.py
│   ├── Cosine_Similarity.py
│   ├── JSON_converter.py
│   └── ...
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/tigernav-ust/tigernav-ust.github.io.git
cd tigernav-ust.github.io
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔬 Reproducibility

To reproduce the training pipeline:

### 1️⃣ Supervised Fine-Tuning

```bash
python codes/finetuning_trainer.py
```

### 2️⃣ Preference Optimization (ORPO)

```bash
python codes/finetuning_orpo.py
```

### 3️⃣ Evaluation

```bash
python codes/evalualte_model.py
```

### Environment Requirements

- Python 3.10+
- CUDA-enabled GPU (recommended ≥8GB VRAM)
- HuggingFace Transformers ecosystem

⚠️ Note: Dataset files are not included due to size and institutional data constraints.

---

## 📖 Citation

If you use TigerNav in your research, please cite:

```bibtex
@inproceedings{tigernav2025,
  title     = {TigerNav: A Campus Navigation Chatbot with Integrated Large Language Model Capability},
  booktitle = {2025 IEEE Region 10 Conference (TENCON)},
  year      = {2025},
  doi       = {10.1109/TENCON66050.2025.11374916}
}
```

---

## 🎓 Affiliation

Department of Electronics Engineering  
Faculty of Engineering  
University of Santo Tomas  
Manila, Philippines  

---

© 2025 TigerNav Research Team • University of Santo Tomas
