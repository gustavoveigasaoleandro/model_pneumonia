# Pneumonia Detection with Deep Learning 🩻🤖

This project uses Deep Learning to detect pneumonia in chest X-ray images using a fine-tuned DenseNet121 model. It also includes a Dash web application that allows users to upload images and view predictions interactively.

---

## 📁 Project Structure

```
.
├── chest_xray/               # X-ray image dataset (not included in repo)
├── models/                   # Saved models (gitignored)
│   └── best_model.keras
├── dash/                     # Dash app components
│   ├── app.py                # Main Dash app
│   ├── components/
│   │   └── UploadForm.py     # Upload form component
│   ├── pages/
│   │   └── index.py          # Main layout
│   └── assets/
│       └── style.css         # Custom CSS
├── model.py                  # Model training script
├── evaluate.py               # Model evaluation script with thresholds
├── .env                      # Environment variables (not committed)
├── .gitignore
└── requirements.txt
```

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/seu-usuario/nome-do-repositorio.git
cd nome-do-repositorio
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Environment Variables

Create a `.env` file in the root directory with the following:

```env
BASE_DIR=./chest_xray
MODEL_SAVE_PATH=./models/model_pneumonia_efficientnet.keras
BEST_MODEL_CHECKPOINT=./models/best_model.keras
```

Ensure the `chest_xray/` dataset folder is properly structured as:

```
chest_xray/
├── train/
├── val/
└── test/
```

---

## 📊 Model Training

Run the full training pipeline:

```bash
python model.py
```

This includes:

- Data augmentation  
- Class balancing  
- Transfer learning with DenseNet121  
- Fine-tuning  
- Checkpoint saving  

---

## 🔍 Model Evaluation

To evaluate model performance using various thresholds:

```bash
python evaluate.py
```

You will receive:

- Accuracy, Precision, Recall, F1-Score  
- Confusion Matrix for thresholds 0.3 to 0.7  

---

## 🌐 Dash Application

To run the Dash app:

```bash
cd dash
python app.py
```

Upload an X-ray image and get instant prediction (Pneumonia or Normal).

---

## 📦 Requirements

See `requirements.txt` for all necessary dependencies:

```
tensorflow>=2.10  
numpy  
matplotlib  
scikit-learn  
pandas  
Pillow  
dash  
dash-bootstrap-components  
dotenv  
```

---

## ⚫ .gitignore Highlights

- Ignores the dataset:

```
chest_xray/
```

- Ignores all saved model files:

```
models/*
!models/.gitkeep
```

- Ignores compiled Python caches and virtual environments:

```
__pycache__/
*.pyc
.venv/
venv/
```

---

## 🎉 Credits

Created as part of a Deep Learning study project using public datasets and open-source tools.
