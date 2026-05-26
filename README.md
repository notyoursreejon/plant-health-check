# 🌿 Plant Health Check — AI Plant Disease Diagnostics with Explainability

A full-stack web application that **classifies plant leaf diseases** into **39 categories** using a Convolutional Neural Network, and then **explains _why_ the model made its prediction** using two industry-standard explainability techniques: **LIME** and **SHAP**.

> **Why does this matter?**  
> A model that says _"this leaf has Late Blight"_ is useful — but a model that says _"this leaf has Late Blight **because of the brown lesion area on the lower half**"_ is trustworthy. This project bridges that gap.

---

## ✨ Features

- 🔬 **AI-Powered Diagnosis** — Upload a leaf image and get instant disease classification across 39 classes
- 🧠 **Explainability (XAI)** — LIME superpixel explanations + SHAP Shapley attribution heatmaps
- 📊 **Real-Time Analysis Pipeline** — Background processing with live status polling (prediction → LIME → SHAP)
- 📚 **Disease Knowledge Center** — Searchable, filterable catalog of all 39 plant diseases with symptoms and treatments
- 📈 **Model Analytics Dashboard** — Training curves, confusion matrix, precision-recall curves, architecture details
- 🗄️ **Diagnostic Archive** — Persistent history of all past analyses with full results
- 🎨 **Botanical Precision Design** — Modern Material-inspired UI with Tailwind CSS, Inter font, and micro-animations

---

## 🖥️ Web Application

The project includes a fully functional **FastAPI web application** with 6 pages:

| Page | Route | Description |
|------|-------|-------------|
| **Home** | `/` | Landing page with feature overview and stats |
| **Analysis Terminal** | `/upload` | Drag-and-drop image upload with real-time analysis |
| **Diagnostic Dashboard** | `/results` | AI prediction + LIME/SHAP visualizations |
| **Knowledge Center** | `/knowledge` | Searchable disease catalog (39 diseases, 15 plants) |
| **Diagnostic Archive** | `/history` | Gallery of past analyses |
| **Model Analytics** | `/analytics` | Performance metrics, training curves, confusion matrix |

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/analyze` | Upload image → starts background analysis |
| `GET` | `/api/analyze/{id}/status` | Poll analysis status (prediction → LIME → SHAP) |
| `GET` | `/api/analyze/{id}` | Get full analysis results |
| `GET` | `/api/diseases` | Disease catalog (filterable by `?plant=Tomato`) |
| `GET` | `/api/history` | All past analyses |
| `DELETE` | `/api/history/{id}` | Delete a history entry |
| `GET` | `/api/metrics` | Model performance metrics |

---

## 📸 Results

### Training Performance

| Metric | Value |
|--------|-------|
| Training Accuracy | ~96.9% |
| Validation Accuracy | ~89% |
| Training Loss | 0.096 |
| Validation Loss | 0.44 |
| Classes | 39 |
| Epochs | 5 |
| Parameters | ~11.1M |

<p align="center">
  <img src="training_graph.png" alt="Training Loss and Accuracy" width="600"/>
</p>

### Confusion Matrix (Normalized)

The confusion matrix shows how well the model performs across all 39 disease classes. Values on the diagonal represent correct predictions — the darker the color, the better the accuracy for that class.

<p align="center">
  <img src="confusion_matrix_final.png" alt="Confusion Matrix" width="700"/>
</p>

### Precision-Recall Curves

Each curve represents one disease class. A curve closer to the top-right corner means better performance. The **AP (Average Precision)** score next to each class name summarizes overall performance — closer to 1.0 is better.

<p align="center">
  <img src="precision_recall_curve_multi.png" alt="Precision-Recall Curves" width="700"/>
</p>

### LIME Explanation Example

LIME highlights **which regions of the leaf** the model focuses on to make its prediction. Green boundaries mark areas that **support** the prediction, red areas **contradict** it.

<p align="center">
  <img src="assets/lime_example.png" alt="LIME Explanation" width="800"/>
</p>

---

## 🧠 How It Works

### Step 1 — Classification

A leaf image is fed into a pre-trained CNN model (`leaf_disease_model.h5`). The model outputs probabilities across 39 classes (e.g., `Tomato > Late_blight: 48.7%`, `Potato > Late_blight: 39.2%`).

### Step 2 — Explainability

Two different methods explain the model's decision:

| Method | How It Works | What You See |
|--------|-------------|--------------|
| **LIME** | Hides different parts of the image (~1000 times), observes how the prediction changes, and identifies which regions matter most | Green/red boundary regions on the leaf |
| **SHAP** | Uses Shapley values (from game theory) to calculate the **exact contribution** of each pixel to the prediction | Red/blue heatmap — red pixels push toward the prediction, blue push against it |

**When both methods highlight the same leaf regions**, it confirms the model is learning real disease patterns (e.g., brown lesions, spots) and not shortcuts like background color or lighting.

### Step 3 — Disease Information

The diagnosis is enriched with scientifically accurate disease metadata — symptoms, severity level, treatment recommendations, and category (fungal, bacterial, viral, pest).

---

## 🗂️ Project Structure

```
plant-health-check/
│
├── app.py                            # FastAPI web server (entry point)
├── leaf_disease_model.h5             # Trained CNN model (39 classes)
├── requirements.txt                  # Python dependencies
├── training_graph.png                # Training loss/accuracy plot
├── confusion_matrix_final.png        # Normalized confusion matrix
├── precision_recall_curve_multi.png  # Per-class PR curves
│
├── static/                           # Frontend assets
│   ├── css/
│   │   └── design-system.css         # Custom CSS (animations, badges, toasts)
│   ├── js/
│   │   ├── app.js                    # Shared API client, navigation, utilities
│   │   ├── upload.js                 # Upload terminal logic
│   │   ├── results.js                # Results dashboard rendering
│   │   ├── knowledge.js              # Disease search/filter
│   │   ├── history.js                # History management
│   │   └── analytics.js              # Metrics display
│   ├── index.html                    # Home page
│   ├── upload.html                   # Analysis terminal
│   ├── results.html                  # Diagnostic dashboard
│   ├── knowledge.html                # Disease knowledge center
│   ├── history.html                  # Diagnostic archive
│   └── analytics.html                # Model analytics
│
├── src/
│   ├── config.py                     # Model loading, preprocessing, shared utilities
│   ├── disease_database.py           # Comprehensive 39-class disease catalog
│   ├── lime_explainer.py             # LIME explainability integration
│   ├── shap_explainer.py             # SHAP explainability integration
│   ├── explain.py                    # CLI runner (LIME + SHAP + comparison)
│   ├── generate_metrics.py           # Generates training graph, confusion matrix, PR curves
│   ├── load_dataset.py               # Loads test dataset using ImageDataGenerator
│   └── test_model.py                 # Quick model loading test
│
├── data/
│   ├── test/                         # Test images organized by class (39 folders)
│   ├── uploads/                      # User-uploaded images (webapp)
│   └── history.json                  # Diagnostic session history (webapp)
│
├── outputs/                          # Generated explanation images
│   ├── lime/                         # LIME explanation PNGs
│   ├── shap/                         # SHAP explanation PNGs
│   └── comparison/                   # Side-by-side comparison PNGs
│
└── assets/                           # Images used in README
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repo
git clone https://github.com/notyoursreejon/plant-health-check.git
cd plant-health-check

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Download the Model

Place the trained model file `leaf_disease_model.h5` in the project root directory. Also ensure the test dataset is placed under `data/test/` with class subfolders.

---

## 📖 Usage

### 🌐 Web Application (Recommended)

```bash
# Start the web server
python app.py

# Open in browser
# http://localhost:8000
```

The webapp provides a complete GUI for uploading images, viewing AI predictions with LIME/SHAP explanations, browsing the disease knowledge center, and reviewing past analyses.

### 🖥️ Command-Line Interface

#### Generate Evaluation Metrics

```bash
python src/generate_metrics.py
```

This creates three files:
- `training_graph.png` — Training loss and accuracy curves
- `confusion_matrix_final.png` — Normalized confusion matrix heatmap
- `precision_recall_curve_multi.png` — Per-class precision-recall curves

#### Run Explainability

```bash
# Run both LIME + SHAP (generates comparison too)
python src/explain.py --image "data/test/Tomato___Late_blight/image.jpg"

# LIME only (~30 seconds)
python src/explain.py --image "data/test/Tomato___Late_blight/image.jpg" --lime-only

# SHAP only (~1-5 minutes)
python src/explain.py --image "data/test/Tomato___Late_blight/image.jpg" --shap-only
```

Images auto-open after generation. Outputs are saved to `outputs/lime/`, `outputs/shap/`, `outputs/comparison/`.

#### Quick Tests

```bash
# Test if model loads correctly
python src/test_model.py

# Test if dataset loads correctly
python src/load_dataset.py
```

---

## 🔬 Supported Disease Classes (39)

| Plant | Diseases |
|-------|----------|
| Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| Blueberry | Healthy |
| Cherry | Powdery Mildew, Healthy |
| Corn | Cercospora Leaf Spot (Gray Leaf Spot), Common Rust, Northern Leaf Blight, Healthy |
| Grape | Black Rot, Esca (Black Measles), Leaf Blight (Isariopsis Leaf Spot), Healthy |
| Orange | Haunglongbing (Citrus Greening) |
| Peach | Bacterial Spot, Healthy |
| Pepper Bell | Bacterial Spot, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Raspberry | Healthy |
| Soybean | Healthy |
| Squash | Powdery Mildew |
| Strawberry | Leaf Scorch, Healthy |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |
| Background | Without Leaves |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Backend Framework | FastAPI + Uvicorn |
| Deep Learning | TensorFlow / Keras |
| Model Architecture | CNN (MobileNetV2-based, ~11.1M params) |
| Explainability | LIME, SHAP (Gradient/Deep/Kernel) |
| Frontend | HTML, JavaScript, Tailwind CSS (CDN) |
| UI Icons | Material Symbols Outlined |
| Typography | Inter (Google Fonts) |
| Visualization | Matplotlib, Seaborn |
| Evaluation | scikit-learn |
| Dataset | PlantVillage (39 classes) |
| Language | Python 3.8+ |

---

## 📄 License

This project is for educational and research purposes.

---

## 🙏 Acknowledgements

- [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset) — Open-source plant disease image dataset
- [LIME](https://github.com/marcotcr/lime) — Local Interpretable Model-agnostic Explanations
- [SHAP](https://github.com/shap/shap) — SHapley Additive exPlanations
- [FastAPI](https://fastapi.tiangolo.com/) — Modern Python web framework
- [Tailwind CSS](https://tailwindcss.com/) — Utility-first CSS framework
- TensorFlow / Keras — Deep learning framework
