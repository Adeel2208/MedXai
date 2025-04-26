import os
from uuid import uuid4
from datetime import datetime

from flask import (
    Flask, request, render_template,
    flash, send_from_directory, redirect, url_for
)
from werkzeug.utils import secure_filename

import pydicom
from PIL import Image

# — XAI / model imports —
import torch
import numpy as np
import matplotlib.pyplot as plt
from safetensors.torch import load_file as safe_load
from transformers import ViTForImageClassification, ViTImageProcessor
from captum.attr import IntegratedGradients
from torchvision import transforms

# — AGNO chatbot imports —
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage

# ——— Flask setup ———
app = Flask(__name__)
app.secret_key = "replace-with-your-own-secret"

# ——— Paths & settings ———
UPLOAD_FOLDER       = "uploads"
STATIC_IMAGE_FOLDER = "static/images"
ALLOWED_EXT         = {"dcm", "dicom", "png", "jpg", "jpeg"}
MODEL_SIZE          = (224, 224)

app.config["UPLOAD_FOLDER"]       = UPLOAD_FOLDER
app.config["STATIC_IMAGE_FOLDER"] = STATIC_IMAGE_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_IMAGE_FOLDER, exist_ok=True)

def allowed_file(fn: str) -> bool:
    return "." in fn and fn.rsplit(".", 1)[1].lower() in ALLOWED_EXT

# ——— Load ViT model & prepare transforms ———
MODEL_WEIGHTS = r"C:\Users\Experttech.pk\Desktop\New folder\model.safetensors"
BACKBONE      = "google/vit-base-patch16-224"
device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViTForImageClassification.from_pretrained(
    BACKBONE, num_labels=2, ignore_mismatched_sizes=True
)
state_dict = safe_load(MODEL_WEIGHTS)
model.load_state_dict(state_dict)
model.to(device).eval()

processor = ViTImageProcessor.from_pretrained(BACKBONE)
transform = transforms.Compose([
    transforms.Resize(MODEL_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean,
                         std=processor.image_std)
])

def predict_and_explain(png_path: str, out_name: str):
    """Run inference and generate an IG heatmap overlay."""
    img = Image.open(png_path).convert("RGB")
    inp = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(inp).logits
        pred   = int(logits.argmax(dim=-1).cpu().item())
        label  = "Pneumonia" if pred == 1 else "Normal"

    ig = IntegratedGradients(lambda x: model(x).logits)
    atts, _ = ig.attribute(inputs=inp, target=pred, return_convergence_delta=True, n_steps=50)
    heat = atts.squeeze(0).cpu().numpy()
    heat = np.maximum(heat, 0).sum(axis=0)
    heat = heat / (heat.max() + 1e-10)

    img_np = inp.squeeze(0).cpu().numpy().transpose(1,2,0)
    img_np = img_np * processor.image_std + processor.image_mean
    img_np = np.clip(img_np, 0, 1)

    plt.figure(figsize=(6,6))
    plt.imshow(img_np)
    plt.imshow(heat, cmap="jet", alpha=0.5)
    plt.axis("off")
    save_path = os.path.join(STATIC_IMAGE_FOLDER, out_name)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    return label, out_name

# ——— One‐Shot Medical Analysis Query ———
query = """
You are a highly skilled medical imaging expert with extensive knowledge in radiology and diagnostic imaging. Analyze the medical image and structure your response as follows:

### 1. Image Type & Region
- Identify imaging modality (X-ray/MRI/CT/Ultrasound/etc.).
- Specify anatomical region and positioning.

### 2. Key Findings
- Highlight primary observations systematically.
- Identify potential abnormalities with detailed descriptions.
- Include measurements and densities where relevant.

### 3. Diagnostic Assessment
- Provide primary diagnosis with confidence level.
- List differential diagnoses ranked by likelihood.
- Support each diagnosis with observed evidence.
- Highlight critical/urgent findings.

### 4. Patient-Friendly Explanation
- Simplify findings in clear, non-technical language.
- Avoid medical jargon or provide easy definitions.
- Include relatable visual analogies.

Ensure a structured and medically accurate response using clear markdown formatting.
"""

# ——— Initialize AGNO agent ———
GOOGLE_API_KEY = "AIzaSyA0HWD0PzAgeeSomug41sF3MIrSw-f0Q2k"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
if not GOOGLE_API_KEY:
    raise RuntimeError("Please set your Google API Key")

medical_agent = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[DuckDuckGoTools()],
    markdown=True
)

def analyze_medical_image(image_path: str) -> str:
    """Process and analyze a medical image using the one-shot query."""
    img = Image.open(image_path)
    w, h = img.size
    new_w = 500
    new_h = int(new_w * h / w)
    img_resized = img.resize((new_w, new_h))
    temp_path = os.path.join(UPLOAD_FOLDER, f"tmp_{uuid4().hex}.png")
    img_resized.save(temp_path)

    agno_img = AgnoImage(filepath=temp_path)
    try:
        response = medical_agent.run(query, images=[agno_img])
        return response.content
    except Exception as e:
        return f"⚠️ Analysis error: {e}"
    finally:
        os.remove(temp_path)

# ——— Project & team info ———
team = {
    "lead":       {"name":"Adeel Mukhtar","role":"Team Lead","img":"adeel.jpg"},
    "members":    [
        {"name":"Laiba Eman","role":"Member","img":"laiba.jpg"},
        {"name":"Aisha Rafiq","role":"Member","img":"aisha.jpg"}
    ],
    "supervisor": {"name":"Dr. Muhammad Rehan Chaudhry","role":"Supervisor","img":"rehan.jpg"}
}
intro = (
    "Our FYP “MedXAI” uses transformer-based models and explainable AI "
    "for transparent, interactive medical scan analysis."
)
details = [
    {"title":"Objectives",       "description":"• Multi-planar DICOM viewer\n• Transformer classification\n• XAI overlays\n• Automated reports"},
    {"title":"Methodology",      "description":"1. Normalize images\n2. ViT inference\n3. IG attribution\n4. Render UI"},
    {"title":"Tech Stack",       "description":"• Flask, Python\n• PyTorch, Transformers\n• Captum for XAI\n• React, Bootstrap"},
    {"title":"Expected Outcomes","description":"– Prototype\n– Explainability demo\n– Clinical GUI\n– Full documentation"}
]

# ——— Routes ———
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["GET","POST"])
def upload():
    if request.method == "POST":
        f = request.files.get("dicom_file")
        if not f or not allowed_file(f.filename):
            flash("Please upload a .dcm, .png, .jpg or .jpeg file.")
            return redirect(request.url)

        fn  = secure_filename(f.filename)
        ext = fn.rsplit(".",1)[1].lower()
        uid = uuid4().hex
        tmp = os.path.join(UPLOAD_FOLDER, f"{uid}_{fn}")
        f.save(tmp)

        try:
            if ext in ("dcm","dicom"):
                ds     = pydicom.dcmread(tmp, force=True)
                pilimg = Image.fromarray(ds.pixel_array)
            else:
                with Image.open(tmp) as im:
                    pilimg = im.convert("RGB").copy()
        except Exception as e:
            flash(f"Error reading file: {e}")
            os.remove(tmp)
            return redirect(request.url)

        os.remove(tmp)
        pilimg = pilimg.resize(MODEL_SIZE, Image.LANCZOS)

        # Save input PNG
        png_name = f"{uid}.png"
        png_path = os.path.join(STATIC_IMAGE_FOLDER, png_name)
        pilimg.save(png_path)

        # Run model + XAI
        label, xai_name = predict_and_explain(png_path, f"xai_{uid}.png")
        xai_path        = os.path.join(STATIC_IMAGE_FOLDER, xai_name)

        # Run chatbot analysis
        analysis = analyze_medical_image(xai_path)

        return render_template(
            "view.html",
            img_name = png_name,
            xai_name = xai_name,
            label    = label,
            analysis = analysis,
            now      = datetime.now()
        )

    return render_template("upload.html")

@app.route("/static/images/<filename>")
def serve_image(filename):
    return send_from_directory(STATIC_IMAGE_FOLDER, filename)

@app.route("/about", methods=["GET"])
def about():
    return render_template(
        "about.html",
        team    = team,
        intro   = intro,
        details = details
    )

if __name__ == "__main__":
    app.run(debug=True)
