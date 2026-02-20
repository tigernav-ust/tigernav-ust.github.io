<h1 align="center">🐯 TigerNav</h1>

<p align="center">
  <em>Development of a Virtual Assistant Using an Autoregressive Model for Indoor Navigation</em>
</p>

<p align="center">
  <img alt="TENCON 2025 Accepted" src="https://img.shields.io/badge/IEEE%20TENCON-2025%20Accepted-2ea44f?style=flat">
  <img alt="DOI" src="https://img.shields.io/badge/DOI-10.1109%2FTENCON66050.2025.11374916-2563eb?style=flat">
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

TigerNav is a dialogue-based indoor navigation assistant designed to provide contextualized directional guidance within university buildings. The system leverages an autoregressive Large Language Model (LLM) fine-tuned on structured campus navigation data to interpret diverse user queries and generate coherent, location-aware instructions.

We evaluate multiple training paradigms, including Supervised Fine-Tuning (SFT) and preference-based optimization techniques such as ORPO. Experimental results demonstrate improved semantic alignment and model confidence under preference optimization compared to baseline fine-tuning.

---

## 🗺️ System Overview

TigerNav follows a structured conversational pipeline:

1. **User Query Input**
2. **Intent & Context Processing**
3. **Autoregressive LLM Inference**
4. **Navigation Instruction Generation**

The system is deployed for indoor navigation within the Fr. Roque Ruaño Building at the University of Santo Tomas, supporting both voice and text-based interaction.

---

## 📊 Experimental Results

Model performance was evaluated using:

- **BLEU**
- **METEOR**
- **ROUGE-L**
- **BERTScore**
- **Perplexity**

The General Purpose Trainer consistently outperformed ORPO across most semantic metrics, while ORPO demonstrated stronger alignment behavior in preference-based evaluation scenarios.

For complete quantitative results, please refer to the published paper.

---

## 📁 Repository Structure

```
tigernav-ust.github.io/
│
├── docs/                     # GitHub Pages website
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
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔬 Reproducibility

To reproduce the training and evaluation pipeline:

### 1️⃣ Supervised Fine-Tuning
```bash
python codes/finetuning_trainer.py
```

### 2️⃣ Preference Optimization
```bash
python codes/finetuning_orpo.py
```

### 3️⃣ Evaluation
```bash
python codes/evalualte_model.py
```

### Recommended Environment

- Python 3.10+
- CUDA-enabled GPU (≥ 8GB VRAM recommended)
- HuggingFace Transformers ecosystem

> ⚠️ Dataset files are not included due to institutional and size constraints.

---

## 📖 Citation

If you use TigerNav in your research, please cite:

```bibtex
@inproceedings{sanjuan2025tigernav,
  title     = {TigerNav: Development of a Virtual Assistant Using an Autoregressive Model for Indoor Navigation},
  author    = {San Juan, Ralph Alexander N. and 
               Baetiong, Ernest John Q. and 
               Bantayao, Saranggani J., Jr. and 
               Mangali, Marc Justin M. and 
               Sumo, Carl Kristien P. and 
               Pangaliman, Ma. Madecheen S.},
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

<p align="center">
© 2025 TigerNav Research Team • University of Santo Tomas
</p>
