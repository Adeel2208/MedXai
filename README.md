# MedXai
A Flask-powered web prototype for medical image analysis that uses Vision Transformer models for pneumonia detection, Grad-CAM saliency overlays for explainability, automated PDF reporting, and an integrated AI chatbot for interactive diagnostic insights.
# MedXAI

A Flask-powered web prototype for medical image analysis, combining Vision Transformer models for pneumonia detection, Grad-CAM saliency overlays for explainability, automated PDF reporting, and an integrated AI chatbot for interactive diagnostic insights.

---

## Table of Contents

- [Features](#features)  
- [Demo](#demo)  
- [Architecture](#architecture)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Project Structure](#project-structure)  
- [Model & XAI Details](#model--xai-details)  
- [Chatbot Integration](#chatbot-integration)  
- [License](#license)  

---

## Features

- **DICOM & Standard Image Support**  
  Upload `.dcm`, `.png`, `.jpg`, or `.jpeg` files.  
- **Multi-Panel Viewer**  
  Side-by-side display of original scans and XAI overlays.  
- **Explainable AI**  
  Integrated Gradients heatmaps via Captum.  
- **Automated Reporting**  
  Generate structured PDF summaries.  
- **Interactive Chatbot**  
  AGNO/Gemini-based medical analysis with DuckDuckGo literature search.  
- **Responsive UI**  
  Modern, dark-themed frontend with Bootstrap 5.

---

## Demo

1. Start the server:  
   ```bash
   flask run

2. Browse to http://127.0.0.1:5000/upload

3. Upload a scan → view prediction, overlay, and chatbot analysis on /view

![image](https://github.com/user-attachments/assets/59681aad-594a-41da-8228-2ed1f5efe8ec)

# Clone
git clone https://github.com/<your-username>/MedXAI.git
cd MedXAI

# Virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install
pip install --upgrade pip
pip install -r requirements.txt

Configuration
Place model.safetensors in project root.

Set your Google API key in app.py:

os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"
Usage
bash
Copy
Edit
export FLASK_APP=app.py     # macOS/Linux
set FLASK_APP=app.py        # Windows
flask run
/upload – Upload and analyze a scan

/view – See original image, XAI overlay, prediction, and chatbot report

Project Structure
csharp
Copy
Edit
MedXAI/
├── app.py
├── model.safetensors
├── requirements.txt
├── static/
│   ├── css/
│   │   └── styles.css
│   └── images/
│       ├── ai.jpg
│       └── logo.png
└── templates/
    ├── base.html
    ├── index.html
    ├── upload.html
    ├── view.html
    └── about.html
Model & XAI Details
Base Model: google/vit-base-patch16-224 fine-tuned for pneumonia (2 classes).

XAI: Integrated Gradients (Captum) → per-pixel saliency maps.

Chatbot Integration
Agent: AGNO Gemini(id="gemini-2.0-flash-exp")

Prompt: One-shot medical analysis (modality, findings, diagnosis, patient-friendly explainer, references).

Tools: DuckDuckGo for recent literature search.

Future Improvements
Multi-slice DICOM support

Generate & download PDF reports

User authentication & audit trails

Dockerized deployment

Support CT/MRI modalities

Enhanced chatbot context handling

