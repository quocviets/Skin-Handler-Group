import os
import cv2
import pandas as pd
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog

# ========== CONFIG MODEL ==========
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13
cfg.MODEL.WEIGHTS = "model_0044999.pth"  # Thay đường dẫn đúng nếu cần
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.15
cfg.MODEL.DEVICE = "cpu"
predictor = DefaultPredictor(cfg)

# ========== CLASS MAPPING ==========
class_id_to_name = {
    3: "mun_dau_den",
    4: "u_nang",
    10: "mun_dau_trang"
}
custom_metadata = MetadataCatalog.get("skin_infer")
custom_metadata.set(thing_classes=list(class_id_to_name.values()))

# ========== PATHS ==========
input_dir = r"C:/Users/lequo/Downloads/Skin_detect/folder_test"
output_dir = r"C:/Users/lequo/Downloads/Skin_detect/test_output"
os.makedirs(output_dir, exist_ok=True)

# ========== PROCESS ==========
results = []

for file_name in os.listdir(input_dir):
    if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(input_dir, file_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"⚠️ Không thể đọc ảnh: {file_name}")
            continue

        outputs = predictor(image)
        instances = outputs["instances"].to("cpu")

        pred_boxes = instances.pred_boxes if instances.has("pred_boxes") else []
        pred_scores = instances.scores if instances.has("scores") else []
        pred_classes = instances.pred_classes if instances.has("pred_classes") else []

        kept_indices = []
        for i in range(len(pred_boxes)):
            class_id = int(pred_classes[i])
            if class_id not in class_id_to_name:
                continue

            score = float(pred_scores[i])
            box = pred_boxes[i].tensor.numpy().tolist()[0]
            class_name = class_id_to_name[class_id]

            results.append({
                "file_name": file_name,
                "class_name": class_name,
                "score": round(score, 4),
                "bbox": [round(x, 2) for x in box]
            })
            kept_indices.append(i)

        # Draw
        if kept_indices:
            for i in kept_indices:
                class_id = int(pred_classes[i])
                class_name = class_id_to_name[class_id]
                score = float(pred_scores[i])
                box = [int(x) for x in pred_boxes[i].tensor.numpy()[0]]

                x1, y1, x2, y2 = box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)

                label_text = f"{class_name} {score:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                font_thickness = 1
                (w, h), _ = cv2.getTextSize(label_text, font, font_scale, font_thickness)
                cv2.rectangle(image, (x1, y1 - h - 4), (x1 + w, y1), (0, 0, 0), -1)
                cv2.putText(image, label_text, (x1, y1 - 2), font, font_scale, (255, 255, 255), font_thickness)

            save_path = os.path.join(output_dir, file_name)
            cv2.imwrite(save_path, image)

# ========== EXPORT ==========
df = pd.DataFrame(results)
df.to_csv(os.path.join(output_dir, "detection_results.csv"), index=False)

print("✅ Hoàn tất! Kết quả lưu ở:", output_dir)
