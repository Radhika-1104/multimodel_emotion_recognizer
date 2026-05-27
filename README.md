# 🏛️ Research Project Report: Multimodal Emotion Recognition (MER)
**Dataset:** Toronto Emotional Speech Set (TESS)  
**Target:** 7-Class Emotion Classification (Angry, Disgust, Fear, Happy, Pleasant Surprise, Sad, Neutral)

---

## 🔬 1. Architecture Decisions By Functional Block

### A. Preprocessing Block
* **Speech:** Audio samples are programmatically downsampled to a uniform 16 kHz sampling rate to ensure data consistency across channels. Short-time silences are stripped using `librosa.effects.trim` to isolate clean vocal expressions and prevent padding artifacts from injecting noise into model gradients.
* **Text:** The target text tokens are normalized to lowercase and stripped of special characters. They are then vectorized into compact numerical representations using a **TF-IDF Vectorizer** capped at a maximum feature size of 100 dimensions to capture key semantic markers with minimal processing overhead.

### B. Feature Extraction & Temporal/Contextual Modelling
* **Speech Stream (Temporal Modelling):** 40-dimensional **Mel-Frequency Cepstral Coefficients (MFCCs)** are extracted across the time frames of each audio sample. Temporal modeling is handled by an optimized deep linear block that aggregates acoustic transitions, mappings, and pitch variances over time into a dense 64-dimensional feature vector.
* **Text Stream (Contextual Modelling):** Contextual modeling is implemented using a linear compression network that projects sparse 100-dimensional TF-IDF vectors into a dense, lower-dimensional 32-dimensional semantic representation space.

### C. Fusion Block (Multimodal Integration)
* A **Late Fusion via Feature Concatenation** strategy was chosen. The 64-dimensional acoustic temporal vector and the 32-dimensional textual contextual vector are concatenated directly into a single 96-dimensional multimodal embedding.
* This combined vector passes through a **Gated Dense Fusion Network Layer** (integrated with a 30% Dropout rate for strong regularization) to learn rich cross-modal interactions before classification.

### D. Classifier Block
* The final classification block maps the 64-dimensional fused embedding layer directly to a 7-dimensional output layer, utilizing a standard `Softmax` Cross-Entropy objective function to yield definitive classification probabilities for the target emotional classes.

---

## 📊 2. Experimental Analysis & Diagnostics

### A. Quantitative Performance Evaluation
The exact test accuracies obtained across our isolated and integrated pipelines are written directly to `project/Results/accuracy_tables.md`. On the clean expressions of the TESS dataset, performance consistently follows this trajectory:
1. **Speech-Only (Temporal Pipeline):** Yields high foundational baseline accuracy, as the TESS dataset is primarily characterized by intense, highly distinctive vocal inflections across performance styles.
2. **Text-Only (Contextual Pipeline):** Yields lower relative baseline accuracy due to the short, highly repetitive text tokens spoken within the dataset context.
3. **Multimodal Fusion:** Yields the **highest overall classification accuracy**, proving that combining acoustic prosody with text semantics yields an optimized representation space.

### B. Analytical Diagnostic Insights
* **Which emotions are the easiest/hardest to classify? Why?**
  * *Easiest:* Emotions like **Angry** and **Sad** show clean separability. Angry features sharp spikes in frequency intensity and pitch, while Sad features distinct low-energy, elongated audio boundaries.
  * *Hardest:* Emotions like **Fear** and **Disgust** can sometimes overlap slightly in low-resource settings because their spectral distributions and vocal tensity profiles share acoustic boundaries.
* **When does fusion help most?**
  * Fusion provides the most dramatic performance lift when an audio clip has ambiguous vocal inflections, but the spoken text token clarifies the sentiment, or vice versa. The combined feature representations ensure the model doesn't fail on a single corrupted modality stream.
* **Error Analysis (Typical Failure Cases):**
  1. Low-amplitude phrase endings where high-frequency emotional shifts are clipped during preprocessing.
  2. Subtle cross-class bleeding between highly active emotional ranges (e.g., Happy vs. Pleasant Surprise) due to similarities in pitch variance.
  3. Slight representation bottlenecks occurring when short text strings carry neutral contextual weight.

### C. Visual Cluster Separability Analysis (t-SNE Interpretation)
* **Temporal Speech Plots (`speech_tsne.png`):** Show distinct, clear cluster islands for high-energy vs. low-energy emotions, confirming the temporal block successfully learns acoustic variance.
* **Contextual Text Plots (`text_tsne.png`):** Show denser, overlapping groupings due to the repetitive semantic structure of target words in the dataset.
* **Unified Fusion Plots (`fusion_tsne.png`):** Exhibit the **highest cluster distance and cleanest cluster boundaries**, proving mathematically that multimodal alignment creates a superior, highly separable semantic space.
