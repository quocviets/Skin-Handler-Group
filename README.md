# 🧠 Skin Handler: Facial Skin Problem Detection

Đây là dự án AI sử dụng mô hình Detectron2 và YOLOv7 để phát hiện các vấn đề da liễu trên khuôn mặt như mụn, nám, u nang, vết thâm,... Dự án bao gồm các thành phần chính sau:

---

## 📁 Cấu trúc thư mục

```bash
.
├── folder_test/              # Thư mục chứa ảnh test đầu vào
├── test_output/              # Thư mục xuất ảnh kết quả + file CSV
├── model_0044999.pth         # File mô hình đã huấn luyện Detectron2 (13 lớp)
├── test.py                   # Script chạy detect và vẽ kết quả
├── Skin_Handler_Train.ipynb  # Notebook huấn luyện model Detectron2
├── .gitattributes            # Dùng để hỗ trợ Git LFS (cho file lớn)
└── README.md                 # Tài liệu mô tả này
```

---

## 🛠 Môi trường cài đặt

- Python ≥ 3.8
- Detectron2 ≥ 0.6
- OpenCV
- PyTorch
- pandas
- Git LFS (để tải mô hình `.pth`)

Cài nhanh Detectron2:

```bash
pip install opencv-python pandas
pip install torch torchvision torchaudio
pip install git+https://github.com/facebookresearch/detectron2.git
```

---

## 🚀 Cách sử dụng

### 1. Test ảnh bằng `test.py`

- Đảm bảo `model_0044999.pth` có trong thư mục chính.
- Chạy file `test.py` để xử lý tất cả ảnh trong `folder_test`:

```bash
python test.py
```

Kết quả:
- Ảnh có khung sẽ nằm trong `test_output`
- Chi tiết phát hiện sẽ ghi vào file CSV: `detection_results.csv`

### 2. Huấn luyện lại mô hình

Mở `Skin_Handler_Train.ipynb` để:

- Re-train từ đầu
- Tiếp tục huấn luyện từ checkpoint (`model_0044999.pth`)
- Theo dõi loss + đánh giá theo từng epoch

---

## 🏷 Các lớp mô hình hỗ trợ

Mô hình Detectron2 được huấn luyện với 13 lớp sau:

- Dark Circle
- Melasma
- PIH
- Blackhead
- Cyst
- Freckles
- Nodule
- Papule
- Pustule
- Skin Pore
- Whitehead
- Wrinkle
- Melasma-Acne-Wrinkle

---

## 📦 Ghi chú

- Mô hình `.pth` lớn hơn 100MB → sử dụng **Git LFS** để quản lý.
- Cập nhật `MetadataCatalog` trong code nếu bạn thay đổi thứ tự lớp.

---

## ✨ Credits

Dự án thuộc nhóm **Skin Handler** – tập trung phát triển giải pháp AI hỗ trợ chăm sóc và chẩn đoán da liễu.
