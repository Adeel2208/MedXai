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

---

## Usage

1. Ensure you have placed your `model.safetensors` in the project root.  
2. In `app.py`, set your Google API key:

   ```python
   os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"

MedXAI/
├── app.py                  # Main application script
├── model.safetensors       # Pre-trained model weights
├── requirements.txt        # Project dependencies
├── static/                 # Static assets
│   ├── css/
│   │   └── styles.css      # CSS styles
│   └── images/
│       ├── ai.jpg          # AI-related imagery
│       ├── logo.png        # Project logo
│       └── architecture.png # Model architecture diagram
└── templates/              # HTML templates
├── base.html           # Base template
├── index.html          # Homepage
├── upload.html         # Image upload page
├── view.html           # Image analysis view
└── about.html          # About page


# MedXAI

A medical imaging analysis platform leveraging AI for pneumonia detection with explainable AI (XAI) and chatbot integration.

## Project Structure

```plaintext
MedXAI/
├── app.py                  # Main application script
├── model.safetensors       # Pre-trained model weights
├── requirements.txt        # Project dependencies
├── static/                 # Static assets
│   ├── css/
│   │   └── styles.css      # CSS styles
│   └── images/
│       ├── ai.jpg          # AI-related imagery
│       ├── logo.png        # Project logo
│       └── architecture.png # Model architecture diagram
└── templates/              # HTML templates
    ├── base.html           # Base template
    ├── index.html          # Homepage
    ├── upload.html         # Image upload page
    ├── view.html           # Image analysis view
    └── about.html          # About page

## Model & XAI Details

### Base Model
- **Model**: Google Vision Transformer (`google/vit-base-patch16-224`)
- **Fine-tuning**: Trained for binary classification (Normal vs. Pneumonia)

### Explainability
- **Method**: Integrated Gradients (via Captum)
- **Output**: Pixel-level saliency maps highlighting regions contributing to predictions

### Preprocessing
- **Image Size**: Resize to 224×224 pixels
- **Normalization**: Adjust to model's expected mean and standard deviation

## Chatbot Integration

### Agent
- **Model**: AGNO Gemini (`gemini-2.0-flash-exp`)

### Prompt Workflow
1. **Image Modality & Region**: Identify imaging type and anatomical focus
2. **Key Findings**: Summarize critical observations
3. **Diagnostic Assessment**: Provide AI-driven diagnosis
4. **Patient-Friendly Explanation**: Simplify findings for non-experts
5. **Research Context**: Incorporate recent medical literature (via DuckDuckGo)

### Tools
- **Search**: DuckDuckGo for up-to-date medical literature

## Future Improvements
- 🖼️ Support for multi-slice 3D DICOM series
- 📋 PDF report generation and download
- 🔒 User authentication and audit trail
- 🐳 Dockerized deployment for scalability
- 🧠 Extended support for CT and MRI modalities
- 🔄 Enhanced chatbot session memory for improved context retention
