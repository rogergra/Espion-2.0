from flask import Flask, request, render_template, jsonify, send_from_directory
import cv2
import numpy as np
import os
import pytesseract
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configurer le chemin de Tesseract si nécessaire
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def detect_and_repair(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Détection des barres (version améliorée)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    masks = [
        cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255])),  # Rouge
        cv2.inRange(hsv, np.array([20, 50, 50]), np.array([40, 255, 255])), # Jaune
        cv2.inRange(hsv, np.array([100, 50, 50]), np.array([140, 255, 255])) # Bleu
    ]
    combined_mask = sum(masks)
    
    # 2. Extraction du texte sous les barres
    text_roi = cv2.bitwise_and(gray, gray, mask=255-combined_mask)
    text = pytesseract.image_to_string(Image.fromarray(text_roi), lang='fra+eng')
    
    # 3. Réparation de l'image
    repaired = cv2.inpaint(img, combined_mask, 7, cv2.INPAINT_NS)
    
    return {
        "text": text.strip(),
        "mask": combined_mask,
        "repaired": repaired
    }

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['image']
        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            try:
                result = detect_and_repair(filepath)
                
                # Sauvegarde des résultats
                result_path = os.path.join(app.config['UPLOAD_FOLDER'], f"result_{file.filename}")
                cv2.imwrite(result_path, result["repaired"])
                
                return jsonify({
                    "success": True,
                    "text": result["text"],
                    "result_image": f"result_{file.filename}"
                })
                
            except Exception as e:
                return jsonify({"success": False, "error": str(e)})
    
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def send_result(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)