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

## ⚙️ Backend Architecture (Python / FastAPI)

The backend is built with **FastAPI** and handles model inference, explainability generation, data persistence, and static file serving. Here is a step-by-step breakdown of every backend component:

### Step 1 — Server Initialization (`app.py`)

When `python app.py` is executed, the following happens in order:

1. **Directory Setup** — Creates required directories if missing: `static/`, `static/css/`, `static/js/`, `data/uploads/`, `outputs/`
2. **FastAPI App Creation** — Initializes the FastAPI instance with CORS middleware (allows all origins for local development)
3. **Static File Mounts** — Three `StaticFiles` mounts serve assets to the browser:
   - `/static` → `static/` (HTML, CSS, JS files)
   - `/outputs` → `outputs/` (generated LIME/SHAP explanation images)
   - `/uploads` → `data/uploads/` (user-uploaded leaf images)
4. **Root Image Routes** — Serves training metrics images (`training_graph.png`, `confusion_matrix_final.png`, `precision_recall_curve_multi.png`) from the project root at `/images/{filename}`
5. **HTML Page Routes** — Six `GET` routes serve HTML pages: `/`, `/upload`, `/results`, `/knowledge`, `/history`, `/analytics`
6. **GPU Configuration** — Calls `configure_gpu()` from `src/config.py` to set TensorFlow memory growth
7. **Model Pre-Loading** — Calls `load_model()` at startup so the first analysis request doesn't wait for model loading (~5-10 seconds saved)
8. **History File Init** — Creates `data/history.json` with an empty array `[]` if it doesn't exist

### Step 2 — ML Pipeline (`src/config.py`)

The core ML pipeline provides these functions used throughout the backend:

| Function | What It Does |
|----------|--------------|
| `configure_gpu()` | Sets TensorFlow GPU memory growth to prevent OOM errors |
| `load_model()` | Loads `leaf_disease_model.h5` (MobileNetV2-based CNN, ~11.1M params). Cached after first call |
| `preprocess_image(path)` | Reads image → resizes to 224×224 → normalizes to [0,1] → returns shape `(1, 224, 224, 3)` |
| `load_image_for_display(path)` | Same as above but returns `(224, 224, 3)` without batch dimension (for LIME/SHAP) |
| `predict_fn(images)` | Runs batch inference → returns probability array `(N, 39)` |
| `get_prediction_summary(path)` | Full pipeline: preprocess → predict → returns `(predicted_class, confidence, top_5_results)` |
| `get_background_images(n)` | Samples `n` random images from `data/test/` for SHAP background dataset |
| `ensure_output_dirs()` | Creates `outputs/lime/`, `outputs/shap/`, `outputs/comparison/` |

### Step 3 — Disease Database (`src/disease_database.py`)

A comprehensive Python dictionary containing scientifically accurate metadata for all **39 disease classes**. Each entry includes:

```python
"Tomato___Late_blight": {
    "plant": "Tomato",
    "disease": "Late Blight",
    "display_name": "Tomato — Late Blight",
    "description": "Caused by Phytophthora infestans...",
    "symptoms": ["Dark water-soaked lesions...", "White fuzzy growth...", ...],
    "treatment": ["Apply copper-based fungicide...", "Remove infected plants...", ...],
    "severity": "high",       # none | low | moderate | high
    "category": "fungal"      # healthy | fungal | bacterial | viral | pest | other
}
```

Exported functions: `get_disease_info(class_name)`, `get_all_diseases()`, `get_diseases_by_plant(plant)`, `get_unique_plants()`

### Step 4 — Analysis Pipeline (Background Worker)

When a user uploads an image via `POST /api/analyze`, the backend follows this pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                    POST /api/analyze                            │
│  1. Validate file type (must be image/*)                        │
│  2. Generate unique job ID (UUID, first 8 chars)                │
│  3. Save uploaded file to data/uploads/{id}.jpg                 │
│  4. Create in-memory job entry (status: "processing")           │
│  5. Start background thread → run_analysis_worker()             │
│  6. Return immediately: { id, status: "processing" }            │
└─────────────┬───────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│              BACKGROUND THREAD: run_analysis_worker()           │
│                                                                 │
│  Phase 1 — Prediction (~1 second)                               │
│    • get_prediction_summary() → class, confidence, top-5        │
│    • get_disease_info() → symptoms, treatment, severity         │
│    • Update job status → "prediction_done"                      │
│                                                                 │
│  Phase 2 — LIME Explanation (~10-30 seconds)                    │
│    • PlantLIMEExplainer().visualize(img, num_samples=800)        │
│    • Generates 4-panel image: original + positive superpixels   │
│      + pros/cons + heatmap                                      │
│    • Saved to outputs/lime/{id}_lime_explanation.png             │
│    • Update job status → "lime_done"                            │
│                                                                 │
│  Phase 3 — SHAP Explanation (~1-5 minutes)                      │
│    • PlantSHAPExplainer(n_background=25).visualize(img)          │
│    • Auto-selects: GradientExplainer → DeepExplainer → Kernel   │
│    • Generates 3-panel image: original + heatmap + bar chart    │
│    • Saved to outputs/shap/{id}_shap_explanation.png             │
│    • Update job status → "complete"                             │
│                                                                 │
│  Phase 4 — Persist to History                                   │
│    • Append full result to data/history.json (thread-safe lock) │
│    • Keep max 100 entries (oldest trimmed)                       │
└─────────────────────────────────────────────────────────────────┘
```

The frontend polls `GET /api/analyze/{id}/status` every 3 seconds. Each status update includes all available data so far, enabling progressive rendering.

### Step 5 — Data Persistence

| Storage | Location | Format | Thread Safety |
|---------|----------|--------|---------------|
| Analysis jobs (active) | In-memory `dict` | Python dict | `threading.Lock` |
| Analysis history | `data/history.json` | JSON array (max 100) | `threading.Lock` |
| Uploaded images | `data/uploads/` | Original image files | Filesystem |
| LIME outputs | `outputs/lime/` | PNG (4-panel, ~450KB) | Filesystem |
| SHAP outputs | `outputs/shap/` | PNG (3-panel, ~800KB) | Filesystem |

### Step 6 — LIME Explainer (`src/lime_explainer.py`)

The `PlantLIMEExplainer` class generates superpixel-based explanations:

1. **Segmentation** — Image is divided into superpixels (contiguous regions)
2. **Perturbation** — ~800 versions of the image are created by randomly hiding/showing superpixels
3. **Inference** — All perturbed images are fed through the model
4. **Linear Model** — A simple linear model is fit to learn which superpixels contribute most
5. **Visualization** — 4-panel output:
   - Panel 1: Original image with top-3 predictions
   - Panel 2: Positive superpixels (green boundaries — support prediction)
   - Panel 3: Positive + negative superpixels (green = support, red = contradict)
   - Panel 4: Continuous importance heatmap overlay (RdYlGn colormap)

### Step 7 — SHAP Explainer (`src/shap_explainer.py`)

The `PlantSHAPExplainer` class generates Shapley value attribution maps with automatic fallback:

1. **Background Selection** — 25 random images from the test set serve as the baseline
2. **Explainer Selection** (automatic fallback chain):
   - `GradientExplainer` — fastest, uses model gradients (default)
   - `DeepExplainer` — DeepLIFT-based, exact attributions (fallback 1)
   - `KernelExplainer` — model-agnostic, slowest, runs on 64×64 downscaled images (fallback 2)
3. **SHAP Value Computation** — Calculates per-pixel attribution for the predicted class
4. **Visualization** — 3-panel output:
   - Panel 1: Original image with top-3 predictions
   - Panel 2: Signed SHAP heatmap (red = supports prediction, blue = opposes)
   - Panel 3: Top-5 class mean SHAP values bar chart

---

## 🎨 Frontend Architecture (HTML / JavaScript / Tailwind CSS)

The frontend is a **multi-page application** (MPA) — each page is a standalone HTML file styled with **Tailwind CSS** (CDN) and powered by vanilla **JavaScript** modules. No build step or bundler is required.

### Design System — "Botanical Precision"

The UI follows a custom Material Design 3-inspired design system:

| Token | Value | Usage |
|-------|-------|-------|
| `primary` | `#0d631b` | Primary actions, links, active states |
| `primary-container` | `#2e7d32` | Button backgrounds, hero accents |
| `secondary` | `#286b33` | Secondary actions |
| `tertiary` | `#005a8c` | Information, analytics highlights |
| `error` | `#ba1a1a` | Error states, high severity badges |
| `background` | `#f8faf8` | Page background (subtle green tint) |
| `surface-container-lowest` | `#ffffff` | Card backgrounds |
| `outline-variant` | `#bfcaba` | Borders, dividers |
| Font | Inter (400–900) | All text via Google Fonts CDN |
| Icons | Material Symbols Outlined | 24px, variable fill/weight |

### Custom CSS (`static/css/design-system.css`)

Extends Tailwind with custom utilities:

| Class | Purpose |
|-------|---------|
| `.shadow-ambient` | Subtle green-tinted card shadow |
| `.animate-fade-in` | Fade + slide-up entry animation |
| `.animate-slide-up` | Stronger slide-up for hero elements |
| `.animate-pulse-gentle` | Gentle opacity pulse for loading states |
| `.animate-spin-slow` | 2s rotation for spinner icons |
| `.skeleton` | Shimmer loading placeholder (gradient animation) |
| `.toast` / `.toast.show` | Fixed-position toast with slide-up transition |
| `.chip-high/medium/low` | Confidence level badges (green/amber/red) |
| `.severity-high/moderate/low/none` | Disease severity badges |
| `.tag-fungal/bacterial/viral/pest/healthy` | Disease category color tags |
| `.icon-fill` | Material Symbols filled variant |

### Shared JavaScript (`static/js/app.js`)

Every page loads `app.js` which provides:

| Export | Description |
|--------|-------------|
| `API.get(url)` | Fetch wrapper with JSON parsing and error handling |
| `API.post(url, body)` | POST with FormData support |
| `API.delete(url)` | DELETE request |
| `showToast(msg, type)` | Animated toast notification (success/error/info) |
| `setActiveNav()` | Highlights current page in sidebar/mobile nav |
| `formatConfidence(val)` | `0.487` → `"48.7%"` |
| `getConfidenceClass(val)` | Returns `chip-high/medium/low` CSS class |
| `formatClassName(cls)` | `"Tomato___Late_blight"` → `"Tomato — Late blight"` |
| `formatTimestamp(ts)` | ISO string → `"May 26, 2026, 07:13 AM"` |
| `getSeverityClass(sev)` | Returns `severity-high/moderate/low/none` CSS class |
| `getCategoryClass(cat)` | Returns `tag-fungal/bacterial/...` CSS class |

### Navigation Layout

**Desktop (≥768px):** Fixed left sidebar (256px wide) with:
- Logo + "Plant Health Check" + "Diagnostic Terminal" header
- Nav links: Home, Analysis, Metrics, Library, Archive (with Material icons)
- Active state: secondary-container background with filled icon
- Bottom: "New Analysis" button

**Mobile (<768px):** Sticky top bar with:
- Logo + app name
- Horizontally scrollable tab bar below
- Active tab: primary color + bottom border

### Page-by-Page Frontend Breakdown

#### 1. Home Page (`index.html`)

| Section | Content |
|---------|---------|
| **Top Nav** | Separate from sidebar — horizontal nav bar with logo + page links |
| **Hero** | Large headline "AI-Powered Plant Disease Detection", subtitle, CTA button → `/upload` |
| **XAI Feature Grid** | 2×2 bento grid explaining LIME, SHAP, disease database, model metrics |
| **Stats Bar** | 39 disease classes • 15 plant species • 96.9% training accuracy • 11.1M parameters |
| **Footer** | Copyright + attribution |

#### 2. Upload Page (`upload.html` + `upload.js`)

| Step | What Happens |
|------|-------------|
| **Drag & Drop Zone** | User drags/clicks to select a leaf image (accepts `image/*`) |
| **File Preview** | Shows thumbnail, filename, file size, format badge |
| **Quality Check** | Animated checklist: ✓ Format Valid → ✓ Resolution OK → ✓ Size Acceptable |
| **Submit** | "Begin Analysis" button sends `POST /api/analyze` with `FormData` |
| **Redirect** | On success, navigates to `/results?id={job_id}` |

**JavaScript flow (`upload.js`):**
```
User drops file → FileReader.readAsDataURL() → show preview
  → click "Begin Analysis" → new FormData() → API.post('/api/analyze', formData)
  → receive { id } → window.location = '/results?id=' + id
```

#### 3. Results Page (`results.html` + `results.js`)

| Phase | UI State | Data Source |
|-------|----------|-------------|
| **Loading** | Skeleton placeholders + "Processing..." status | — |
| **Prediction Ready** | Disease name, confidence bar, severity badge, top-5 predictions list | `status: "prediction_done"` |
| **LIME Ready** | LIME 4-panel image loads in the explanation panel | `status: "lime_done"` |
| **SHAP Ready** | SHAP 3-panel image loads, "Complete" badge shown | `status: "complete"` |
| **Error** | Error message with retry suggestion | `status: "error"` |

**JavaScript flow (`results.js`):**
```
Page load → extract ?id= from URL
  → GET /api/analyze/{id}/status (every 3 seconds)
  → progressively render: prediction → disease info → LIME image → SHAP image
  → stop polling when status === "complete" or "error"
```

**Key UI components:**
- Original image display with zoom
- Confidence percentage bar (color-coded: green ≥90%, amber ≥70%, red <70%)
- Disease info card: description, symptoms list, treatment list, severity + category badges
- LIME panel: loads `lime_image_url` when available
- SHAP panel: loads `shap_image_url` when available
- Top-5 predictions table with probability bars

#### 4. Knowledge Page (`knowledge.html` + `knowledge.js`)

| Feature | Implementation |
|---------|---------------|
| **Search** | Text input filters diseases by name, plant, description, symptoms, treatment |
| **Plant Filter** | Dropdown populated from `GET /api/diseases` → `plants[]` array |
| **Stats Bar** | Shows total diseases, total plants, filtered count |
| **Disease Cards** | Responsive grid (1/2/3 columns). Each card shows: icon, disease name, plant, severity badge, category tag |
| **Expand/Collapse** | Click a card → reveals description, symptoms list, treatment list (animated) |

**JavaScript flow (`knowledge.js`):**
```
Page load → GET /api/diseases → populate plant dropdown + render grid
  → search input 'input' event → filter diseases array → re-render
  → plant dropdown 'change' event → filter diseases array → re-render
  → card click → toggle .hidden on .disease-details div
```

#### 5. History Page (`history.html` + `history.js`)

| Feature | Implementation |
|---------|---------------|
| **Gallery Grid** | Responsive card grid (1/2/3 columns) showing thumbnail, disease name, confidence, timestamp |
| **Click to View** | Clicking a card navigates to `/results?id={id}` |
| **Delete** | Hover reveals delete button → confirmation dialog → `DELETE /api/history/{id}` → refresh grid |
| **Empty State** | Shows icon + "No diagnostics yet" + CTA button → `/upload` |

#### 6. Analytics Page (`analytics.html` + `analytics.js`)

| Section | Data Source |
|---------|-------------|
| **Metric Cards** (4) | Training accuracy, validation accuracy, training loss, validation loss from `GET /api/metrics` |
| **Model Architecture** | Architecture name, total params (formatted as "11.1M"), classes, epochs |
| **Training Curves** | `<img>` loading `/images/training_graph.png` with graceful fallback |
| **Confusion Matrix** | `<img>` loading `/images/confusion_matrix_final.png` |
| **PR Curves** | `<img>` loading `/images/precision_recall_curve_multi.png` |
| **Disease Table** | Full 39-row table: #, Plant, Disease, Category tag, Severity badge from `GET /api/diseases` |

### Frontend → Backend Data Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                         BROWSER                                  │
│                                                                  │
│  upload.js ──POST /api/analyze──────────┐                        │
│                                         │                        │
│  results.js ─GET /api/analyze/{id}/status (polling every 3s)     │
│       │                                 │                        │
│       ├─ renders prediction immediately │                        │
│       ├─ loads LIME image when ready    │                        │
│       └─ loads SHAP image when ready    │                        │
│                                         ▼                        │
│  knowledge.js ─GET /api/diseases ──→ renders searchable grid     │
│  history.js ───GET /api/history ───→ renders gallery + delete    │
│  analytics.js ─GET /api/metrics ───→ renders metric cards        │
└──────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                     FASTAPI SERVER (app.py)                       │
│                                                                  │
│  /api/analyze ──→ save file → start thread → return job ID       │
│  /api/analyze/{id}/status ──→ read from analysis_jobs dict       │
│  /api/diseases ──→ disease_database.py → filter by plant         │
│  /api/history ──→ read data/history.json                         │
│  /api/metrics ──→ load model params + hardcoded training stats   │
│  /static/* ──→ serve HTML/CSS/JS                                 │
│  /outputs/* ──→ serve LIME/SHAP generated images                 │
│  /uploads/* ──→ serve user-uploaded images                       │
└──────────────────────────────────────────────────────────────────┘
```

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
