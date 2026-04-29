# 📸 Image Captioning with CNN + RNN — Flickr8k

An end-to-end **Image Captioning** system that generates natural language descriptions for images by combining:

- **Computer Vision (CNN)** — ResNet-50 for visual feature extraction
- **Natural Language Processing (RNN)** — GRU/LSTM decoders for text generation

Trained and evaluated on the **Flickr8k** dataset using an NVIDIA H100 80GB GPU.

---

## 📁 Project Structure

```
├── 01_eda.ipynb              # Task 1: Data exploration & preprocessing
├── 02_modelling.ipynb        # Tasks 2–4: Model building, training & improvement
├── 03_model_analysis.ipynb   # Tasks 5–6: Evaluation, visualization & error analysis
├── models.py                 # Shared model definitions & utilities
├── artifacts/                # Processed data artifacts & training results
│   ├── eda_artifacts.pkl
│   └── training_results.pkl
├── saved_models/             # Trained model checkpoints (not tracked by git)
│   ├── baseline_gru.pth
│   ├── baseline_lstm.pth
│   ├── gru_attn.pth
│   └── lstm_attn.pth
├── flickr8k/                 # Dataset (not tracked by git — >1GB)
├── report.md                 # Project report
└── README.md                 # This file
```

---

## 🧪 Models Trained

| Model              | Best Val Loss | Training Time |
|---------------------|---------------|---------------|
| Baseline GRU        | **2.8385**    | 607.3 s       |
| Baseline LSTM       | 2.9302        | 604.8 s       |
| GRU + Attention     | 2.9037        | 607.9 s       |
| LSTM + Attention    | 2.9894        | 609.7 s       |

### Embedding Size Ablation (GRU baseline, 5 epochs)

| Embedding Dim | Best Val Loss |
|---------------|---------------|
| 128           | 3.1352        |
| 256           | 3.0994        |
| **512**       | **3.0304**    |

---

## 🏗️ Architecture

### Baseline (GRU / LSTM)
```
Image → ResNet50 (frozen) → Linear → h₀
Caption[:-1] → Embedding → GRU/LSTM(h₀) → Dense → Predicted tokens
```

### Enhanced (with Bahdanau Attention)
```
Image → ResNet50 (frozen) → Linear → (B, 49, D)  [spatial features]
For each timestep t:
    context = BahdanauAttention(features, h_{t-1})
    h_t     = GRUCell/LSTMCell([embed(word_t) || context], h_{t-1})
    out_t   = Dense(h_t)
```

---

## ⚙️ Hyperparameters

| Parameter      | Value  |
|----------------|--------|
| Embedding Dim  | 256    |
| Hidden Units   | 256    |
| Dropout        | 0.4    |
| Batch Size     | 512    |
| Epochs         | 15     |
| Learning Rate  | 1e-3   |
| Mixed Precision| bfloat16 (H100) |

---

## 📊 Dataset

- **Flickr8k**: 8,091 images, each with 5 captions (40,455 total)
- **Splits**: ~80% train / ~10% validation / ~10% test
- **Vocabulary**: 2,988 words (frequency threshold ≥ 5)
- **Max sequence length**: 26 tokens

---

## 🚀 How to Reproduce

1. **Clone the repository** (includes trained model checkpoints via Git LFS)
   ```bash
   git lfs install
   git clone https://github.com/hamza-hesham-hendy/Deeplearning_Project_-image_captioning-.git
   cd Deeplearning_Project_-image_captioning-
   ```

2. **Install dependencies**
   ```bash
   pip install torch torchvision numpy pandas matplotlib seaborn wordcloud tqdm nltk wandb pillow
   ```

3. **Run evaluation directly**
   Since pretrained model checkpoints are included in `saved_models/`, you can skip training and go straight to:
   - `03_model_analysis.ipynb` — Evaluate, visualize, and analyze errors

4. *(Optional)* **Retrain from scratch**
   If you want to reproduce the full pipeline, configure your Kaggle API credentials and run all notebooks in order:
   ```bash
   pip install kaggle
   # Place your kaggle.json in ~/.kaggle/ (Linux/Mac) or %USERPROFILE%\.kaggle\ (Windows)
   ```
   - `01_eda.ipynb` — Automatically downloads Flickr8k, preprocesses data and saves artifacts
   - `02_modelling.ipynb` — Train all models

---

## 🔑 Key Findings

- **CNNs for images**: CNNs excel at extracting hierarchical spatial features (edges → textures → objects) through convolutional filters with weight sharing, making them ideal image encoders.
- **RNNs for text**: LSTM/GRU networks model sequential dependencies in language, maintaining memory of previously generated words to produce coherent captions.
- **GRU vs LSTM**: The simpler GRU baseline achieved the best validation loss (2.8385), likely due to better generalization on the small Flickr8k dataset.
- **Attention mechanism**: Bahdanau attention enables the decoder to focus on relevant spatial regions at each timestep, providing interpretability through attention maps (Grad-CAM).
- **Overfitting**: Attention models showed mild overfitting after ~10 epochs (train loss continuing to decrease while validation loss plateaued), addressed with dropout (0.4) and early stopping.

---