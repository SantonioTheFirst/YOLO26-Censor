import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
from ultralytics import YOLO

# --- Константы и настройки ---
FACE_MODEL_PATH = "yolo26n-face.pt"
NSFW_MODEL_PATH = "yolo26n-nsfw.pt"

# --- 1. Кэширование моделей (Чистая функция без st.* элементов) ---
@st.cache_resource(show_spinner="Загрузка нейросетей в память...")
def load_ai_models():
    results = {"models": {}, "status": {}}
    
    # Загрузка модели лиц
    try:
        if os.path.exists(FACE_MODEL_PATH):
            results["models"]["face"] = YOLO(FACE_MODEL_PATH)
            results["status"]["face"] = "custom"
        else:
            # Фолбэк на стандартную модель, если кастомная не найдена
            results["models"]["face"] = YOLO("yolov8n.pt")
            results["status"]["face"] = "fallback"
    except Exception as e:
        results["status"]["face"] = f"error: {str(e)}"

    # Загрузка модели NSFW
    try:
        if os.path.exists(NSFW_MODEL_PATH):
            results["models"]["nsfw"] = YOLO(NSFW_MODEL_PATH)
            results["status"]["nsfw"] = "ready"
        else:
            results["status"]["nsfw"] = "missing"
    except Exception as e:
        results["status"]["nsfw"] = f"error: {str(e)}"
        
    return results

# --- 2. Алгоритмы цензуры ---
def apply_pixelate(roi, intensity):
    """Пикселизация: intensity (0.1 - 1.0), где 0.1 - очень крупные пиксели."""
    h, w = roi.shape[:2]
    if h == 0 or w == 0: return roi
    # Вычисляем количество блоков: чем меньше intensity, тем меньше блоков (крупнее пиксель)
    blocks = int(max(2, 50 * intensity)) 
    temp = cv2.resize(roi, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

def apply_blur(roi, intensity):
    """Размытие по Гауссу."""
    h, w = roi.shape[:2]
    if h == 0 or w == 0: return roi
    k_size = int((intensity * min(h, w)) // 2) * 2 + 1
    return cv2.GaussianBlur(roi, (k_size, k_size), 0) if k_size > 1 else roi

def apply_solid(roi):
    """Сплошная заливка серым."""
    return np.full_like(roi, (64, 64, 64))

# --- 3. Главная функция обработки ---
def process_frame(image, models_dict, targets, conf, mode, intensity):
    img_array = np.array(image) # PIL (RGB) -> NumPy (RGB)
    h, w = img_array.shape[:2]
    all_boxes = []

    # Сбор рамок от модели лиц
    if "Лица" in targets and "face" in models_dict:
        # Для кастомной модели лиц класс обычно 0. Для yolov8n класс person тоже 0.
        res = models_dict["face"](img_array, conf=conf, verbose=False)
        for r in res:
            for box in r.boxes:
                if int(box.cls[0]) == 0:
                    all_boxes.append(map(int, box.xyxy[0].tolist()))

    # Сбор рамок от модели NSFW
    if "NSFW" in targets and "nsfw" in models_dict:
        res = models_dict["nsfw"](img_array, conf=conf, verbose=False)
        for r in res:
            for box in r.boxes:
                # Здесь нужно указать индексы классов твоей NSFW модели
                # Например: [1, 2, 3] для разных типов контента
                all_boxes.append(map(int, box.xyxy[0].tolist()))

    # Применение выбранного метода ко всем рамкам
    for x1, y1, x2, y2 in all_boxes:
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1: continue
        
        roi = img_array[y1:y2, x1:x2]
        
        if mode == "Pixelate":
            img_array[y1:y2, x1:x2] = apply_pixelate(roi, intensity)
        elif mode == "Blur":
            img_array[y1:y2, x1:x2] = apply_blur(roi, intensity)
        elif mode == "Solid":
            img_array[y1:y2, x1:x2] = apply_solid(roi)

    return Image.fromarray(img_array)

# --- 4. Интерфейс Streamlit ---
def main():
    st.set_page_config(page_title="AI Censor Tool", layout="wide")
    st.title("🛡️ Universal AI Censor")

    # Загрузка моделей
    data = load_ai_models()
    models = data["models"]
    status = data["status"]

    # Sidebar: Настройки
    with st.sidebar:
        st.header("Настройки")
        targets = st.multiselect("Что скрывать?", ["Лица", "NSFW"], default=["Лица"])
        conf_val = st.slider("Confidence threshold", 0.1, 1.0, 0.4)
        
        st.divider()
        mode = st.radio("Метод цензуры", ["Pixelate", "Blur", "Solid"])
        intensity = st.slider("Интенсивность", 0.1, 1.0, 0.3) if mode != "Solid" else 1.0

        # Уведомления о статусе моделей
        if status["face"] == "fallback":
            st.warning("⚠️ Файл yolo26n-face.pt не найден. Используется базовая модель.")
        if status["nsfw"] == "missing" and "NSFW" in targets:
            st.error("❌ Модель NSFW не загружена (нет файла .pt)")

    # Main Area: Загрузка
    uploaded_file = st.file_uploader("Загрузите фото", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        input_img = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Оригинал")
            st.image(input_img, use_container_width=True)
            
        with col2:
            st.subheader("Результат")
            with st.spinner("Нейросеть работает..."):
                output_img = process_frame(input_img, models, targets, conf_val, mode, intensity)
                st.image(output_img, use_container_width=True)
                
                # Кнопка скачивания
                buf = io.BytesIO()
                output_img.save(buf, format="JPEG", quality=90)
                st.download_button("⬇️ Скачать результат", buf.getvalue(), f"censored_{uploaded_file.name}", "image/jpeg")

if __name__ == "__main__":
    main()
    
