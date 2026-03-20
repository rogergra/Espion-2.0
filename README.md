# Espion-2.0

# ESPION 2.0 - Forensic Image Reveal Tool

**Outil avancé de récupération de texte caché, censure, marker et redaction sur images et documents.**

Développé par **Roger Amouzou**  
**Togo** 🇹🇬  
**Admin Réseau & Chercheur en Cybersécurité**  
**Contact :** rogashack@gmail.com  
**GitHub :** [@rogergra](https://github.com/rogergra)

### 🎯 Fonctionnalités
- Reveal texte barré / masqué / marker noir (Blue Channel Inversion + EasyOCR)
- Détection et suppression de censure / blackout / pixelation
- Extraction EXIF + métadonnées cachées
- Inpaint avancé (TELEA)
- Génération automatique de **rapport PDF forensic** (images avant/après + SHA-256 + texte)
- Interface web hacker style (Matrix / Terminal)

### 🚀 Installation rapide

```bash
git clone https://github.com/rogergra/Espion-2.0.git
cd Espion-2.0
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 app.py

```
Puis, ouvrez **http://127.0.0.1:5000** dans votre navigateur.

## 📦 Technologies utilisées
- Python
- Flask (interface web)
- OpenCV (traitement d'image)
- Tesseract OCR (reconnaissance de texte)  lien du telechargement : https://sourceforge.net/projects/tesseract-ocr.mirror
- ESRGAN (restauration d'images)

## 📜 Licence
MIT License

Copyright (c) 2026 Roger Amouzou (Togo)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
## 📬 Contact
Si vous avez des questions ou des suggestions, contactez-moi via [GitHub](https://github.com/rogergra), (rogashack@gmail.com).
bye me a cofee rogashack@gmail.com

