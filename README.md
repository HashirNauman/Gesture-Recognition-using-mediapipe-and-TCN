# Video Pipeline for Gesture-Based Pronunciation Assessment

## Overview
This module focuses on the **video-based gesture analysis pipeline** used for evaluating articulation patterns in pronunciation learning. It processes input videos, extracts skeletal keypoints, and performs gesture similarity analysis to assess correctness.

The design prioritizes:
- Low latency (real-time usability)
- Privacy (no raw video storage)
- Efficiency (small dataset compatibility)

---

## Problem Statement
Traditional pronunciation systems rely heavily on audio, ignoring **visual articulation cues** such as lip and hand movements.

This pipeline addresses:
- Gesture correctness evaluation
- Real-time feedback constraints
- Small dataset limitations
- Privacy concerns in video processing

---

## Pipeline Architecture

```
Input Video
   ↓
Frame Extraction (OpenCV)
   ↓
Keypoint Detection (MediaPipe Holistic)
   ↓
Sequence Preprocessing
   ↓
Temporal Modeling (TCN)
   ↓
Embedding Generation
   ↓
Similarity Matching (Cosine Similarity)
   ↓
Top-K Gesture Predictions
```

---

## Data Processing

### Keypoint Extraction
We use **MediaPipe Holistic** to extract:
- 33 Pose landmarks
- 21 Left-hand landmarks
- 21 Right-hand landmarks

Each frame produces:
```
225-dimensional feature vector
```

### Preprocessing Steps
- Normalize keypoints to [0,1]
- Handle missing frames (zero-padding)
- Fix sequence length (100 frames)
- Remove low-quality samples

This ensures stable input across varying video conditions.

---

## Model Architecture

### Temporal Convolutional Network (TCN)

Instead of LSTMs or Transformers, we use **TCN**:

- 3 Temporal Blocks
- Kernel size: 7
- Dropout: 0.2
- Output: 64-dimensional embedding

### Why TCN?
- Parallel computation → faster inference  
- Better suited for small datasets  
- Lower latency than recurrent models  

---

## Gesture Matching

Instead of pure classification, the system uses:

**Embedding-based similarity**

- Input embedding compared with reference embeddings
- Cosine similarity used for scoring
- Returns **Top-3 closest gestures**

This makes results:
- More interpretable
- More robust in low-data scenarios

---

## Training Details

- Dataset size: ~350 videos
- Classes: 7 gestures
- Train/Val/Test split: ~292 / 62 / 64
- Optimizer: Adam
- Learning rate: 5e-4
- Loss: Cross-entropy

---

## Results

| Model              | Test Accuracy |
|-------------------|--------------|
| TCN (Final)       | **81–85%**   |
| TCN-CNN Hybrid    | 67–70%       |
| LSTM              | 61–67%       |
| Transformer       | ~60%         |

### Key Observations
- TCN performs best due to efficient temporal modeling
- LSTM suffers from high latency
- Transformers underperform due to limited data
- Confusion exists between visually similar gestures

---

## Optimization for Real-Time Use

- Cached keypoints (.npy) to avoid recomputation
- Frame downscaling → ~30% speed boost
- Reduced channels → faster inference with minor accuracy tradeoff

---

## Privacy Considerations

- No raw video storage
- Only skeletal keypoints retained
- Reduces risk of identity leakage

---

## Future Improvements

- Larger dataset for better generalization
- Improved handling of similar gestures
- Model quantization for edge deployment
- Multimodal fusion with audio pipeline

---

## Integration

```
Video Input → Video Pipeline → Gesture Score
                             ↘
                              Combined with Audio Score
```

---

## Note

This repository contains a **high-level implementation overview only**.  
Detailed model configurations, dataset specifics, and proprietary components are intentionally omitted.
