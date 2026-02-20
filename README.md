<h1 align="center">🐯 TigerNav</h1>

<p align="center">
  <em>TigerNav: Development of a Virtual Assistant Using an Autoregressive Model for Indoor Navigation</em>
</p>

<p align="center">
  <img alt="IEEE TENCON 2025 Accepted" src="https://img.shields.io/badge/IEEE%20TENCON-2025%20Accepted-2ea44f?style=flat">
  <img alt="DOI" src="https://img.shields.io/badge/DOI-10.1109%2FTENCON66050.2025.11374916-2563eb?style=flat">
  <img alt="License MIT" src="https://img.shields.io/badge/License-MIT-f59e0b?style=flat">
</p>

<p align="center">
  📄 <a href="https://doi.org/10.1109/TENCON66050.2025.11374916"><strong>Paper</strong></a>
  &nbsp;•&nbsp;
  🌐 <a href="https://tigernav-ust.github.io/"><strong>Project Page</strong></a>
  &nbsp;•&nbsp;
  💻 <a href="https://github.com/tigernav-ust/tigernav-ust.github.io/tree/main/docs/codes"><strong>Code</strong></a>
  &nbsp;•&nbsp;
  📋 <a href="#citation"><strong>BibTeX</strong></a>
</p>

---

## ✨ Overview

TigerNav is a dialogue-based indoor navigation virtual assistant designed to provide contextualized directional guidance inside university buildings. It leverages an autoregressive language model fine-tuned on structured indoor navigation data to interpret diverse user queries and generate coherent, location-aware instructions.

---

## 🗺️ System Overview

**Pipeline**
1. **User Query Input** (voice or text)
2. **Speech Processing** (for voice mode)
3. **Autoregressive Model Inference**
4. **Navigation Instruction Generation**
5. **Speech Output** (for voice mode)

<p align="center">
  <img src="docs/architecture.png" alt="TigerNav System Architecture" width="900">
</p>

> If your images are stored directly under `/docs/`, keep the path as `docs/filename.png`.
> If they are in another folder (e.g., `assets/`), just update the paths above.

---

## 💻 User Interface

<p align="center">
  <img src="docs/gui-voice.png" alt="Voice UI" width="440">
  &nbsp;&nbsp;
  <img src="docs/gui-chat.png" alt="Chat UI" width="440">
</p>

---

## 📊 Results (Summary)

Evaluation was conducted using:
- **BLEU**
- **METEOR**
- **ROUGE-L**
- **BERTScore**
- **Perplexity**

For full quantitative results and experimental setup, please refer to the published paper.

---

## 📁 Repository Structure
