import cv2, pytesseract
class Vision:
    def __init__(self, cfg): self.lang = cfg["vision"]["ocr_lang"]
    def ocr(self, img_path: str) -> str:
        img = cv2.imread(img_path)
        if img is None: return ""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return pytesseract.image_to_string(gray, lang=self.lang)
