import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
import logging
from ultralytics import YOLO

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Настройка страницы ---
st.set_page_config(
    page_title="AI Privacy & Censor Tool",
    page_icon="🛡️",
    layout="wide"
)

# --- 1. Инициализация и Кэширование Моделей ---
# @st.cache_resource гарантирует, что модели загрузятся 1 раз и будут висеть в памяти
@st.cache_resource(show_spinner="Загрузка моделей YOLO в память...")
def load_models():
    models = {}
    try:
        # Пытаемся загрузить модели. 
        # В реальном проекте здесь лежат yolo26n-face.pt и yolo26n-nsfw.pt
        if os.path.exists("yolo26n-face.pt"):
            models['face'] = YOLO("yolo26n-face.pt")
        else:
            # Фолбэк для MVP, если кастомной модели нет (используем стандартную)
            models['face'] = YOLO("yolov8n.pt") 
            st.toast("Модель лиц не найдена. Используется стандартная YOLOv8n (класс person).", icon="⚠️")

        if os.path.exists("yolo26n-nsfw.pt"):
            models['nsfw'] = YOLO("yolo26n-nsfw.pt")
            
        return models
    except Exception as e:
        st.error(f"Ошибка загрузки моделей: {e}")
        return {}

# --- 2. Алгоритмы цензуры ---
def apply_gaussian_blur(roi: np.ndarray, ratio: float) -> np.ndarray:
    h, w = roi.shape[:2]
    if h == 0 or w == 0: return roi
    min_dim = min(h, w)
    k = int((ratio * min_dim) / 2) * 2 + 1
    return cv2.GaussianBlur(roi, (k, k), 0) if k > 1 else roi

def apply_pixelate(roi: np.ndarray, blocks: int) -> np.ndarray:
    h, w = roi.shape[:2]
    if h == 0 or w == 0: return roi
    # Защита от слишком большого количества блоков
    blocks = max(1, min(blocks, min(w, h)))
    temp = cv2.resize(roi, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

def apply_solid_fill(roi: np.ndarray) -> np.ndarray:
    return np.full_like(roi, (50, 50, 50)) # Темно-серый цвет в RGB

# --- 3. Основная логика обработки ---
def process_image(image: Image.Image, models: dict, targets: list, conf: float, mode: str, intensity: float) -> Image.Image:
    """Обрабатывает изображение: детекция -> цензура -> возврат результата."""
    
    # PIL Image (RGB) -> NumPy Array (RGB)
    img_array = np.array(image)
    
    boxes_to_censor = []

    # Детекция лиц
    if "Лица" in targets and 'face' in models:
        # Если используем стандартную YOLO, класс person = 0
        face_cls = 0 
        results = models['face'](img_array, conf=conf, verbose=False)
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == face_cls:
                    boxes_to_censor.append(map(int, box.xyxy[0].tolist()))

    # Детекция NSFW
    if "NSFW" in targets and 'nsfw' in models:
        nsfw_classes = [1, 2, 4] # Индексы классов NSFW
        results = models['nsfw'](img_array, conf=conf, verbose=False)
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) in nsfw_classes:
                    boxes_to_censor.append(map(int, box.xyxy[0].tolist()))

    # Применение цензуры (работаем с NumPy array in-place)
    for x1, y1, x2, y2 in boxes_to_censor:
        h, w = img_array.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if (x2 - x1) <= 0 or (y2 - y1) <= 0: continue
        
        roi = img_array[y1:y2, x1:x2]
        
        if mode == "Размытие (Blur)":
            img_array[y1:y2, x1:x2] = apply_gaussian_blur(roi, intensity)
        elif mode == "Пикселизация (Pixelate)":
            # Для пикселизации intensity это количество блоков (инвертируем логику: меньше интенсивность ползунка -> меньше блоков -> крупнее пиксели)
            blocks = int(max(5, 50 * (1.1 - intensity))) 
            img_array[y1:y2, x1:x2] = apply_pixelate(roi, blocks)
        elif mode == "Сплошная заливка (Solid)":
            img_array[y1:y2, x1:x2] = apply_solid_fill(roi)

    # NumPy Array (RGB) -> PIL Image
    return Image.fromarray(img_array)

# --- 4. Веб-интерфейс (UI) ---
def main():
    st.title("🛡️ AI Privacy Censor")
    st.markdown("Инструмент для автоматической анонимизации изображений.")

    # Загружаем модели
    models = load_models()

    # --- Боковая панель (Настройки) ---
    with st.sidebar:
        st.header("⚙️ Настройки детекции")
        
        targets = st.multiselect(
            "Что скрываем?",
            ["Лица", "NSFW"],
            default=["Лица"]
        )
        
        conf_threshold = st.slider(
            "Уверенность нейросети (Confidence)",
            min_value=0.1, max_value=1.0, value=0.5, step=0.05,
            help="Чем выше значение, тем меньше ложных срабатываний, но модель может пропустить некоторые объекты."
        )

        st.divider()
        st.header("🎨 Настройки цензуры")
        
        mode = st.radio(
            "Режим работы",
            ["Пикселизация (Pixelate)", "Размытие (Blur)", "Сплошная заливка (Solid)"]
        )
        
        # Ползунок интенсивности меняет смысл в зависимости от режима
        intensity = 0.5
        if mode != "Сплошная заливка (Solid)":
            intensity = st.slider(
                "Интенсивность",
                min_value=0.1, max_value=1.0, value=0.5, step=0.1,
                help="Для размытия — сила фильтра. Для пикселизации — размер квадратов."
            )

    # --- Основной экран (Загрузка и результат) ---
    uploaded_file = st.file_uploader("Загрузи изображение (JPG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Читаем изображение в PIL
        image = Image.open(uploaded_file).convert('RGB')
        
        # Создаем две колонки для сравнения
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Оригинал")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Результат")
            
            if not targets:
                st.info("Выбери объекты для скрытия в левом меню.")
                st.image(image, use_container_width=True)
            else:
                with st.spinner("Обработка нейросетью..."):
                    # Вызываем функцию обработки
                    processed_image = process_image(image, models, targets, conf_threshold, mode, intensity)
                    st.image(processed_image, use_container_width=True)

                # --- Кнопка скачивания ---
                # Конвертируем PIL Image обратно в байты для скачивания
                buf = io.BytesIO()
                processed_image.save(buf, format="JPEG", quality=90)
                byte_im = buf.getvalue()

                st.download_button(
                    label="⬇️ Скачать результат",
                    data=byte_im,
                    file_name=f"censored_{uploaded_file.name}",
                    mime="image/jpeg",
                    use_container_width=True
                )

if __name__ == "__main__":
    main()
