# Image Captioning with CNN + RNN — Project Report

---

## 1. Problem Description

Image captioning is the task of automatically generating a natural language description for a given image. It sits at the intersection of **Computer Vision** and **Natural Language Processing**, requiring a system that can both *understand* visual content and *produce* coherent textual descriptions.

This project implements an encoder–decoder image captioning pipeline on the **Flickr8k** dataset. The encoder uses a pretrained **ResNet-50** CNN to extract visual features, while the decoder uses recurrent neural networks (**GRU** and **LSTM**) to generate word sequences. We further enhance the baseline with **Bahdanau (additive) attention**, which allows the decoder to focus on relevant spatial regions of the image at each generation step.

The system is trained with **teacher forcing** and evaluated using BLEU scores, Grad-CAM visualizations, and detailed error analysis.

---

## 2. Approach

### 2.1 Data Preparation (Task 1)

**Dataset**: Flickr8k — 8,091 images, each paired with 5 human-written captions (40,455 image–caption pairs total).

**Image preprocessing**:
- Resized to 224×224 pixels
- Normalized using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

**Text preprocessing**:
- Lowercased all captions
- Removed punctuation and special characters
- Added `<start>` and `<end>` tokens to each caption
- Built a vocabulary with frequency threshold ≥ 5 (words appearing fewer than 5 times mapped to `<unk>`)
- Final vocabulary size: **2,988 words**
- Padded/truncated all sequences to a maximum length of **26 tokens**

**Data splits**: ~80% training / ~10% validation / ~10% test (810 test images, 4,050 test captions).

### 2.2 Exploratory Data Analysis

Key findings from the EDA notebook:
- Captions are typically 8–15 words long, with a right-skewed distribution
- The most frequent words are common nouns and verbs (e.g., "dog", "man", "running", "playing")
- Word cloud analysis reveals the dataset is dominated by outdoor/action scenes
- Images are diverse in content but share common themes (people, dogs, outdoor activities)

---

## 3. Model Architecture

### 3.1 Baseline Model (Tasks 2 & 3)

The baseline follows the classic **encoder–decoder** paradigm:

```
Image Encoder:
  Image → ResNet-50 (frozen, pretrained on ImageNet) → Global Avg Pool → Linear(2048 → 256) → h₀

Text Decoder:
  Caption[:-1] → Embedding(vocab_size, 256) → Dropout(0.4) → GRU/LSTM(256, hidden=256) → Dense(vocab_size)
```

**Why CNN for images?** Convolutional Neural Networks are the gold standard for visual feature extraction because:
- They learn hierarchical representations (edges → textures → parts → objects)
- Weight sharing in convolutional filters makes them translation-invariant
- Pretrained CNNs (trained on millions of ImageNet images) provide rich, transferable features
- They compress high-dimensional pixel data into compact, semantically meaningful feature vectors

**Why LSTM/GRU for text generation?** Recurrent networks are well-suited for sequential text generation because:
- They maintain a hidden state that captures context from previously generated words
- Gating mechanisms (forget/update gates in LSTM; reset/update gates in GRU) help manage long-range dependencies
- They naturally model the autoregressive nature of language (each word depends on prior words)
- Teacher forcing during training provides stable gradient signals

### 3.2 Enhanced Model with Attention (Task 4)

The attention-based model replaces the single global feature vector with spatial features and a learned attention mechanism:

```
Image Encoder:
  Image → ResNet-50 (frozen) → Conv features (7×7×2048) → Linear → (B, 49, 256) [spatial features]

Bahdanau Attention:
  For each timestep t:
    energy = V · tanh(W₁ · features + W₂ · h_{t-1})
    α = softmax(energy)
    context = Σ(α · features)

Text Decoder:
  [embed(word_t) || context] → GRUCell/LSTMCell(512, 256) → Dense(vocab_size)
```

This allows the model to dynamically attend to different spatial regions when generating each word, providing interpretability through attention weight visualization.

### 3.3 Model Improvements (Task 4)

The following improvements were explored:
- **Dropout (0.4)**: Applied to embedding and decoder layers to regularize and prevent overfitting
- **Embedding size ablation** (128 / 256 / 512): Larger embeddings (512) achieved the best validation loss in controlled 5-epoch experiments
- **GRU vs LSTM comparison**: Both architectures were trained with and without attention
- **Mixed precision training** (bfloat16): Leveraged H100 GPU capabilities for faster training
- **Learning rate scheduling**: ReduceLROnPlateau used for attention models
- **Beam search decoding**: Implemented alongside greedy decoding for improved inference quality

---

## 4. Experiments and Results

### 4.1 Training Configuration

| Parameter        | Value           |
|------------------|-----------------|
| Image Encoder    | ResNet-50 (frozen, ImageNet pretrained) |
| Embedding Dim    | 256             |
| Hidden Units     | 256             |
| Dropout          | 0.4             |
| Batch Size       | 512             |
| Optimizer        | Adam            |
| Learning Rate    | 1e-3            |
| Epochs           | 15              |
| Mixed Precision  | bfloat16        |
| Hardware         | NVIDIA H100 80GB HBM3 |

### 4.2 Training Results

| Model               | Best Val Loss | Final Train Loss | Training Time |
|----------------------|---------------|------------------|---------------|
| **Baseline GRU**     | **2.8385**    | 2.3806           | 607.3 s       |
| Baseline LSTM        | 2.9302        | 2.6114           | 604.8 s       |
| GRU + Attention      | 2.9037        | 2.2326           | 607.9 s       |
| LSTM + Attention     | 2.9894        | 2.5373           | 609.7 s       |

**Key observations**:
- The **Baseline GRU** achieved the lowest validation loss (2.8385), suggesting that for the Flickr8k dataset scale, the simpler architecture generalizes better
- Attention models achieved lower *training* losses but higher *validation* losses, indicating mild overfitting
- GRU consistently outperformed LSTM across both baseline and attention variants
- All models trained in approximately 10 minutes on the H100 GPU

### 4.3 Embedding Size Ablation

Using the GRU baseline with 5 training epochs:

| Embedding Dim | Best Val Loss |
|---------------|---------------|
| 128           | 3.1352        |
| 256           | 3.0994        |
| **512**       | **3.0304**    |

Larger embedding dimensions consistently improved performance, suggesting that richer word representations help the model capture more nuanced language patterns.

### 4.4 Loss Curve Analysis

All four models showed consistent convergence patterns:
- Rapid initial descent in both training and validation loss (epochs 1–5)
- Gradual plateau in later epochs (10–15)
- The gap between training and validation loss (overfitting indicator) was most pronounced in attention models
- Baseline models showed healthier convergence with smaller train/val gaps

---

## 5. Error Analysis (Task 6)

### 5.1 Common Failure Modes

Through systematic examination of model predictions versus ground truth captions, we identified the following failure patterns:

**1. Generic/Safe Captions**
The model frequently produces generic descriptions that are technically correct but lack specificity. For example, describing a complex scene as "a dog is running in the grass" when the ground truth mentions specific colors, breeds, or actions. This happens because:
- Common word patterns dominate the training data
- The cross-entropy loss encourages statistically safe predictions
- Teacher forcing during training doesn't penalize lack of diversity

**2. Object Hallucination**
The model sometimes generates references to objects not present in the image. This occurs because:
- The language model component learns strong word co-occurrences (e.g., "dog" often co-occurs with "ball")
- The decoder may override visual evidence with linguistic priors
- Limited visual resolution from the frozen CNN encoder

**3. Missing Objects**
The model may fail to mention important objects or actions visible in the image. Causes include:
- The global average pooling in baseline models compresses spatial information, losing details about smaller or peripheral objects
- Attention models can partially address this but may still miss objects with low attention weights

**4. Repetitive/Redundant Text**
Greedy decoding occasionally produces repetitive phrases (e.g., "a man in a red shirt in a red shirt"). This is a well-known limitation of autoregressive generation and can be partially mitigated by beam search decoding.

**5. Incorrect Actions/Relationships**
The model sometimes misidentifies actions or spatial relationships between objects. For instance, describing someone "sitting" when they are "standing," or confusing "left" and "right" relationships. This reflects the difficulty of grounding verbs and spatial prepositions in visual features.

### 5.2 Why Does the Model Generate Incorrect or Repetitive Captions?

- **Data bias**: The training data is dominated by certain scene types and vocabulary patterns; the model learns to reproduce these biases
- **Teacher forcing gap**: During training, the model always receives the correct previous word; during inference, errors compound as the model conditions on its own (potentially incorrect) predictions
- **Fixed CNN features**: The frozen ResNet-50 encoder was pretrained for classification, not captioning; its features may not encode all visual attributes relevant for description
- **Vocabulary limitations**: Words appearing fewer than 5 times are mapped to `<unk>`, reducing the model's ability to describe unusual objects or actions
- **Small dataset**: Flickr8k (8K images) is relatively small for training a captioning system; larger datasets (e.g., MS-COCO with 330K images) typically yield significantly better results

---

## 6. Limitations of the Model

1. **Small dataset**: Flickr8k contains only ~8K images, limiting the diversity of learned visual-linguistic associations
2. **Frozen encoder**: The CNN is not fine-tuned, so visual features are not optimized for the captioning task
3. **Single-layer decoder**: A deeper decoder (multiple GRU/LSTM layers) could capture more complex language patterns
4. **No scheduled sampling**: Training exclusively with teacher forcing creates a train/test discrepancy
5. **Limited vocabulary**: The frequency threshold filters out rare but potentially important words
6. **No reinforcement learning**: BLEU-optimized training (e.g., SCST — Self-Critical Sequence Training) could directly optimize evaluation metrics
7. **No Transformer architecture**: Modern captioning systems use Transformer-based architectures (e.g., VIT + GPT-style decoder) which handle long-range dependencies more effectively

---

## 7. What Would We Improve with More Time/Resources?

1. **Fine-tune the CNN encoder**: Unfreeze the last few ResNet layers and jointly train encoder + decoder with a lower learning rate
2. **Use a larger dataset**: Train on MS-COCO (330K images, 1.5M captions) for dramatically better performance
3. **Implement Transformer decoder**: Replace GRU/LSTM with a multi-head self-attention decoder for improved long-range dependencies
4. **Scheduled sampling**: Gradually transition from teacher forcing to model's own predictions during training
5. **CIDEr/METEOR metrics**: Evaluate with additional metrics beyond BLEU for more comprehensive quality assessment
6. **Self-Critical Sequence Training (SCST)**: Use reinforcement learning to directly optimize CIDEr scores
7. **Vision Transformer encoder**: Replace ResNet-50 with ViT for patch-based visual feature extraction
8. **Larger embedding/hidden dimensions**: The ablation study shows 512-dim embeddings outperform 256-dim; exploring even larger dimensions with proper regularization could help
9. **Data augmentation**: Apply random crops, flips, color jitter to training images for better generalization
10. **Ensemble methods**: Combine predictions from multiple model variants for improved caption quality

---

## 8. Conclusions

This project successfully implemented an end-to-end image captioning system that combines CNN-based visual feature extraction with RNN-based text generation. Key takeaways include:

- **CNN + RNN is a viable paradigm** for image captioning, even with a small dataset like Flickr8k
- **Simpler models can outperform complex ones** on small datasets — the baseline GRU achieved the best validation loss, highlighting the importance of matching model complexity to data scale
- **Attention mechanisms provide interpretability** through Grad-CAM and attention weight visualizations, even if they don't always improve quantitative metrics
- **Teacher forcing** enables stable training but creates a gap between training and inference that leads to error accumulation
- **Embedding dimensionality matters** — larger word embeddings consistently improved performance in our ablation study

The project demonstrates both the power and limitations of neural image captioning, providing a solid foundation for understanding the challenges at the intersection of computer vision and natural language processing.
