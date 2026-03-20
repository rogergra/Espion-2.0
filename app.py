
"""
ESPION 2.0 - Forensic Image Analysis Tool
Auteur : Roger Amouzou (Togo)
Admin Réseau & Chercheur en Cybersécurité
Mail : rogashack@gmail.com
GitHub : https://github.com/rogergra/Espion-2.0
© 2026 - Tous droits réservés (MIT License)
"""







from flask import Flask, request, render_template, jsonify, send_from_directory, send_file, session
import cv2
import numpy as np
import os
import pytesseract
import easyocr
from PIL import Image as PILImage
import uuid
from datetime import datetime
import hashlib
import io
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# ====================== KILL NNPACK WARNING POUR TOUJOURS ======================
import warnings
warnings.filterwarnings("ignore", message=".*NNPACK.*")
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)
app.secret_key = 'super_secret_key_espion_2026_change_me_later'
app.config['UPLOAD_FOLDER'] = 'upload'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    try:
        # EasyOCR créé ICI (lazy) → plus de warning au démarrage/restart
        reader = easyocr.Reader(['fr', 'en'], gpu=False, verbose=False)

        orig = cv2.imread(image_path)
        if orig is None:
            raise ValueError("Impossible de charger l'image")

        img = orig.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ==================== EXIF RÉEL ====================
        pil_img = PILImage.open(image_path)
        exif_data = {}
        if pil_img._getexif():
            for tag_id, value in pil_img._getexif().items():
                tag = PILImage.ExifTags.get(tag_id, tag_id)
                exif_data[str(tag)] = str(value)
        metadata = exif_data or {"Info": "Aucune EXIF"}

        # ==================== KILLER REVEAL ====================
        config = r'--oem 3 --psm 6 -l fra+eng'

        b, g, r = cv2.split(img)
        blue_inv = 255 - b
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
        super_reveal = clahe.apply(blue_inv)
        blur = cv2.GaussianBlur(super_reveal, (0,0), 3)
        super_reveal = cv2.addWeighted(super_reveal, 2.5, blur, -1.5, 0)

        text_tess = pytesseract.image_to_string(PILImage.fromarray(super_reveal), config=config).strip()
        text_easy = ' '.join(reader.readtext(super_reveal, detail=0, paragraph=True)).strip()

        final_text = text_easy if len(text_easy) > len(text_tess) else text_tess

        # Mask & Inpaint
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0,20,20]), np.array([180,255,255]))
        edges = cv2.Canny(gray, 20, 150)
        mask = cv2.bitwise_or(mask, edges)
        kernel = np.ones((9,9), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        repaired = cv2.inpaint(img, mask, 15, cv2.INPAINT_TELEA)

        # Sauvegarde
        base_name = uuid.uuid4().hex[:12]
        paths = {
            "original": os.path.join(app.config['UPLOAD_FOLDER'], f"orig_{base_name}.jpg"),
            "reveal": os.path.join(app.config['UPLOAD_FOLDER'], f"reveal_{base_name}.jpg"),
            "repaired": os.path.join(app.config['UPLOAD_FOLDER'], f"repaired_{base_name}.jpg"),
            "mask": os.path.join(app.config['UPLOAD_FOLDER'], f"mask_{base_name}.jpg"),
        }

        cv2.imwrite(paths["original"], orig)
        cv2.imwrite(paths["reveal"], super_reveal)
        cv2.imwrite(paths["repaired"], repaired)
        cv2.imwrite(paths["mask"], mask)

        session[base_name] = {
            "text": final_text,
            "filename_orig": os.path.basename(image_path),
            "paths": paths
        }

        return {
            "success": True,
            "text": final_text or "Aucun texte détecté",
            "original": f"orig_{base_name}.jpg",
            "reveal": f"reveal_{base_name}.jpg",
            "repaired": f"repaired_{base_name}.jpg",
            "mask": f"mask_{base_name}.jpg",
            "analysis_id": base_name,
            "metadata": metadata
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

# ====================== PDF ======================
def generate_report_pdf(paths, extracted_text, filename_orig):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=22, textColor=colors.green, spaceAfter=24)
    heading_style = ParagraphStyle('Heading2', parent=styles['Heading2'], fontSize=14, textColor=colors.green, spaceAfter=12)
    mono_style = ParagraphStyle('Mono', parent=styles['Normal'], fontName='Courier', fontSize=10, spaceAfter=8)

    elements = []
    elements.append(Paragraph("ESPION 2.0 - RAPPORT FORENSIC CLASSIFIÉ", title_style))
    elements.append(Paragraph(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    elements.append(Paragraph(f"Fichier source : {filename_orig}", styles['Normal']))

    with open(paths["original"], "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    elements.append(Paragraph(f"SHA-256 : {file_hash}", mono_style))

    img_data = [
        [RLImage(paths["original"], width=170, height=170), RLImage(paths["reveal"], width=170, height=170)],
        [RLImage(paths["repaired"], width=170, height=170), RLImage(paths["mask"], width=170, height=170)]
    ]
    table = Table(img_data, colWidths=[190]*2)
    table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.green)]))
    elements.append(table)

    elements.append(Paragraph("TEXTE EXTRAIT (PAYLOAD RÉVÉLÉ)", heading_style))
    elements.append(Paragraph(extracted_text.replace('\n', '<br/>'), mono_style))
    elements.append(Paragraph("Rapport généré par ROGAS TECH - Usage Red Team / Forensic uniquement.", styles['Normal']))
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ====================== ROUTES ======================
@app.route('/', methods=['GET', 'POST'])
def main_route():
    if request.method == 'POST':
        file = request.files.get('image')
        if not file or file.filename == '' or not allowed_file(file.filename):
            return jsonify({"success": False, "error": "Fichier invalide"})

        ext = file.filename.rsplit('.', 1)[1].lower()
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"input_{uuid.uuid4().hex[:10]}.{ext}")
        file.save(temp_path)

        result = process_image(temp_path)
        return jsonify(result)

    return render_template('upload.html')

@app.route('/upload/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/report/<analysis_id>')
def get_report(analysis_id):
    if analysis_id not in session:
        return "Session expirée. Relance l'analyse.", 404
    data = session[analysis_id]
    pdf_buffer = generate_report_pdf(data["paths"], data["text"], data["filename_orig"])
    return send_file(pdf_buffer, as_attachment=True, download_name=f"ESPION_Forensic_Report_{analysis_id}.pdf", mimetype='application/pdf')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
