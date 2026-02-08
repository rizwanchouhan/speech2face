# 🗣️ Speech2Face: Emotionally Expressive Facial Animation via EmotionBERT Embeddings

<img src="https://i.imgur.com/waxVImv.png" alt="Emotion Chat Inference" style="max-width: 100%;">

## 🧠 Overview

Realistic speech-driven facial animation is essential for building believable digital humans.  
However, many existing approaches struggle to capture the **fine-grained emotional cues** embedded in natural speech, leading to animations that appear neutral or emotionally inconsistent.

We present **Speech2Face**, a multimodal framework that generates high-fidelity facial movements conditioned on expressive speech.

Our key idea:

👉 use **EmotionBERT** to extract rich semantic & emotional representations from transcripts  
👉 combine them with **acoustic features**  
👉 guide a generative model to produce temporally coherent, emotionally aligned facial motion.

Speech2Face improves:

- emotional accuracy  
- naturalness  
- lip-speech synchronization  

across **VOCASET**, **IEMOCAP**, **MEAD**, and **BIWI** benchmarks.

---

## 🏗️ Architecture

The pipeline consists of four major components:

1. **Speech Processing** – extract acoustic descriptors from audio.  
2. **EmotionBERT Encoder** – obtain semantic & emotion embeddings from transcripts.  
3. **Multimodal Fusion** – align acoustic and emotional signals.  
4. **Facial Motion Generator** – synthesize expressive, temporally consistent animations.

<img src="https://github.com/speech2face/blob/main/resources/methodology.jpg" alt="Pipeline" style="max-width: 100%;">

---

## ✨ Key Features

- Emotion-aware speech → face generation  
- Strong semantic conditioning via EmotionBERT  
- Improved temporal stability  
- Works across multiple datasets & speakers  
- Modular training for research flexibility

---

## 📦 Installation

### Prerequisites
- CUDA-enabled GPU  
- Python 3.11  
- Conda  

---

### Setup Environment

```bash
conda create -n speech2face python=3.11 -y
conda activate speech2face
pip install -r requirements.txt
```

---

### Install PyTorch (CUDA example)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

### Known Fix (OmegaConf / antlr)

If dependency conflicts appear:

```bash
pip uninstall omegaconf antlr4-python3-runtime -y
pip install omegaconf==2.3.0 antlr4-python3-runtime==4.9.3
```

---

## 📥 Pretrained Models

```bash
git lfs install
git clone https://huggingface.co/your_repo/speech2face pretrained_models
```

**Note:** Public checkpoints may differ slightly from the results reported in the paper.

---

## 🚀 Quick Start

### 1️⃣ Prepare Data

For simplest usage, organize inputs like this:

```
data/
 ├── videos/   # mp4 files
 └── audios/   # wav files
```

You can change paths inside config files if needed.

---

### 2️⃣ Run Inference (Full Pipeline)

```bash
bash scripts/infer_raw_data.sh \
  --video_dir data/videos \
  --audio_dir data/audios \
  --output_dir outputs \
  --ckpt pretrained_models/model.ckpt
```

This pipeline will:

- compute audio features  
- extract embeddings  
- perform multimodal fusion  
- generate facial animations  

---

### 3️⃣ Advanced / Custom Inference

```bash
bash scripts/inference.sh \
  outputs \
  path/to/filelist.txt \
  path/to/model.ckpt
```

---

## 🏋️ Training

Speech2Face supports modular training and custom datasets.

### Expected Data Layout

```
root/
 ├── videos
 ├── videos_emb
 ├── audios
 └── audios_emb
```

You can modify paths inside the training scripts.

---

### Train the Model

```bash
bash train.sh path/to/filelist.txt [num_workers] [batch_size] [gpus]
```

---

## 📊 Datasets Used

- **VOCASET**  
- **IEMOCAP**  
- **MEAD**  
- **BIWI**

These datasets allow evaluation of emotional fidelity, realism, and synchronization.

---

## 📈 Evaluation Goals

Speech2Face aims to improve:

- emotion correctness  
- perceptual realism  
- motion smoothness  
- audio-visual alignment  

---

## 📁 Project Structure (Simplified)

```
speech2face/
├── configs/
├── datasets/
├── scripts/
├── src/
├── pretrained_models/
├── train.sh
└── README.md
```

---

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@article{speech2face2025,
  title={Speech2Face: Emotionally Expressive Facial Animation via EmotionBERT Embeddings},
  author={Anonymous},
  year={2025}
}
```

---

## 🙏 Acknowledgements

We thank the authors of EmotionBERT and the dataset providers for supporting multimodal affective research.
