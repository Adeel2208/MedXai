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
