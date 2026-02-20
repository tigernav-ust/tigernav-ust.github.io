# 🐯 TigerNav  
### A Campus Navigation Chatbot with Integrated Large Language Model Capability  

[![IEEE TENCON 2025](https://img.shields.io/badge/IEEE%20TENCON-2025-blue)](https://ieeemy.org/tencon2025/)
[![DOI](https://img.shields.io/badge/DOI-10.1109/TENCON66050.2025.11374916-green)](https://doi.org/10.1109/TENCON66050.2025.11374916)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

🌐 **Project Page:** https://tigernav-ust.github.io/  
📄 **Paper (IEEE Xplore):** https://doi.org/10.1109/TENCON66050.2025.11374916  

---

## 📌 Overview

TigerNav is an intelligent campus navigation chatbot designed to assist students, faculty, and visitors in navigating the University of Santo Tomas campus.  

The system integrates:

- 🧠 Large Language Models (LLMs)
- 📚 Fine-tuned domain-specific datasets
- 📊 Preference optimization techniques (SFT, DPO, ORPO)
- 💬 Conversational interface for natural interaction

TigerNav provides contextualized campus directions, building information, and navigation assistance using an optimized instruction-tuned language model.

---

## 🏗️ System Architecture

TigerNav follows a structured pipeline:

1. 📥 User Query  
2. 🔎 Query Processing  
3. 🧠 LLM Inference (Fine-tuned Model)  
4. 📤 Response Generation  

Training strategies evaluated:
- Supervised Fine-Tuning (SFT)
- Direct Preference Optimization (DPO)
- Odds Ratio Preference Optimization (ORPO)

---

## 📊 Results Summary

The evaluation compared multiple training strategies using:

- METEOR (semantic similarity)
- BERTScore
- Perplexity (model confidence)

ORPO demonstrated improved response alignment and confidence compared to baseline SFT.

(See full quantitative results in the published paper.)

---

## 📁 Repository Structure

```
tigernav/
│
├── codes/                     # Sanitized training and evaluation scripts
│   ├── finetuning_trainer.py
│   ├── finetuning_orpo.py
│   ├── DPO_Format.py
│   ├── DPO_dataset_metric.py
│   ├── Cosine_Similarity.py
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

To reproduce training and evaluation:

1. Prepare dataset in required JSON / parquet format.
2. Run supervised fine-tuning:

```bash
python codes/finetuning_trainer.py
```

3. Run preference optimization:

```bash
python codes/finetuning_orpo.py
```

4. Evaluate trained model:

```bash
python codes/evalualte_model.py
```

⚠️ Note:
- Dataset files are not included due to size and institutional constraints.
- Ensure GPU acceleration is available for training.

Recommended environment:
- Python 3.10+
- CUDA-enabled GPU (≥8GB VRAM recommended)

---

## 📄 Citation

If you find TigerNav useful in your research, please cite:

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

## 📜 License

This project is released under the MIT License.

---

© 2025 TigerNav Research Team • University of Santo Tomas
