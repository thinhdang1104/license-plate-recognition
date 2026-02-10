import os
import cv2
import numpy as np
import base64
from ultralytics import YOLO
import streamlit as st


def load_model(path, name):
    if not os.path.exists(path):
        st.warning(f"⚠️ Không tìm thấy {name} model: {path}")
        return None
    return YOLO(path)

PLATE_MODEL = "weights/plate_best.pt"
CHAR_MODEL  = "weights/chars_best.pt"

plate_model = load_model(PLATE_MODEL, "Plate")
char_model  = load_model(CHAR_MODEL, "Character")

st.set_page_config(page_title="Biển số + Ký tự", layout="wide")
st.title("🚗 Nhận diện BIỂN SỐ + KÝ TỰ (YOLOv8)")

conf_plate = 0.40
conf_char  = 0.35

uploaded = st.file_uploader("📸 Chọn ảnh xe", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    results = plate_model.predict(img, conf=conf_plate, iou=0.5, verbose=False)[0]

    if len(results.boxes) == 0:
        st.write("❌ Không phát hiện biển số nào.")
        st.stop()

    st.subheader("🔍 Detect biển số & ký tự:")

    names = char_model.names     
    all_plate_texts = []          

#Vòng lặp
    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 3)
        cv2.putText(
            img,
            f"Plate {i+1}",
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

        crop = img[y1:y2, x1:x2]
        char_res = char_model.predict(
            crop,
            conf=conf_char,
            iou=0.5,
            imgsz=640,
            verbose=False
        )[0]

        if len(char_res.boxes) == 0:
            all_plate_texts.append(f"Biển {i+1}: (không đọc được ký tự)")
            continue

        # ================== LƯU TỌA ĐỘ + NHÃN KÝ TỰ ==================
        chars = []
        for cbox in char_res.boxes:
            cx1, cy1, cx2, cy2 = map(int, cbox.xyxy[0])
            label = names[int(cbox.cls[0])]
            conf  = float(cbox.conf[0])

            x_center = (cx1 + cx2) / 2.0
            y_center = (cy1 + cy2) / 2.0

            chars.append((x_center, y_center, label, conf, (cx1, cy1, cx2, cy2)))

        # ================== TÁCH 2 DÒNG THEO y_center ==================
        y_values = [c[1] for c in chars]
        y_min, y_max = min(y_values), max(y_values)
        threshold = (y_min + y_max) / 2.0   # đường phân chia dòng trên / dòng dưới

        line_top    = [c for c in chars if c[1] <  threshold]
        line_bottom = [c for c in chars if c[1] >= threshold]

        # sort từng dòng theo x_center (trái → phải)
        line_top    = sorted(line_top, key=lambda x: x[0])
        line_bottom = sorted(line_bottom, key=lambda x: x[0])

        # ================== GHÉP CHUỖI BIỂN SỐ ==================
        text_top    = "".join([c[2] for c in line_top])
        text_bottom = "".join([c[2] for c in line_bottom])

        if text_bottom:
            plate_text = f"{text_top}-{text_bottom}"
        else:
            plate_text = text_top

        all_plate_texts.append(f"Biển {i+1}: {plate_text}")

        # ================== VẼ LẠI BOX + LABEL KÝ TỰ LÊN ẢNH GỐC ==================        
        for c in chars:
            _, _, label, conf, (cx1, cy1, cx2, cy2) = c

            cv2.rectangle(
                img,
                (x1 + cx1, y1 + cy1),
                (x1 + cx2, y1 + cy2),
                (255, 0, 0),
                2
            )
            cv2.putText(
                img,
                f"{label} {conf:.2f}",
                (x1 + cx1, y1 + cy1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

        # ================== VẼ CHUỖI BIỂN SỐ DƯỚI KHUNG ==================
        cv2.putText(
            img,
            plate_text,
            (x1, y2 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2
        )

    # ================== HIỂN THỊ ẢNH (THU NHỎ + CĂN GIỮA) ==================
    display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode(".png", display_img)
    img_base64 = base64.b64encode(buffer).decode()

    st.markdown(
        f"""
        <div style="text-align: center;">
            <h3>Kết quả detect biển số + ký tự</h3>
            <img src="data:image/png;base64,{img_base64}" width="650">
        </div>
        """,
        unsafe_allow_html=True
    )

    # ================== HIỂN THỊ TEXT BIỂN SỐ ==================
    st.markdown("### 📃 Kết quả nhận dạng biển số:")
    for txt in all_plate_texts:
        st.write("- ", txt)
