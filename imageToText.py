#EasyOCR ile resimden yazı okunması
import cv2
import easyocr

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Resim yüklenemedi! Lütfen dosya yolunu kontrol edin.")
        return None
    return image

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    denoised = cv2.fastNlMeansDenoising(gray)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    blurred = cv2.GaussianBlur(enhanced, (5, 5), 1.5)
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

    return sharpened

def detect_and_recognize_text(image, reader):
    # EasyOCR ile metin tanıma
    results = reader.readtext(image)

    text_boxes = []
    texts = []

    for (bbox, text, confidence) in results:
        if confidence > 0.4:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            x_min, y_min = map(int, top_left)
            x_max, y_max = map(int, bottom_right)

            text_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))
            texts.append(text)

    return text_boxes, texts

def main():
    # EasyOCR okuyucu (Türkçe ve İngilizce destekli)
    reader = easyocr.Reader(['tr', 'en'])

    image_path = input("Resmin tam yolunu girin: ")
    image = load_image(image_path)
    if image is None:
        return

    processed_image = preprocess_image(image)

    text_boxes, texts = detect_and_recognize_text(processed_image, reader)

    output = image.copy()
    for i, (box, text) in enumerate(zip(text_boxes, texts)):
        x, y, w, h = box

        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

        print(f"Tespit edilen metin {i + 1}: {text}")

        display_text = text[:20] + '...' if len(text) > 20 else text
        cv2.putText(output, display_text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    fixed_width = 600
    fixed_height = 800
    resized_output = cv2.resize(output, (fixed_width, fixed_height))

    cv2.imshow('Tespit Edilen Metinler', resized_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
