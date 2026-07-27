# 🎨 StyleForge AI

> Transform ordinary images into stunning artworks using **Adaptive Instance Normalization (AdaIN)** Neural Style Transfer.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red?style=for-the-badge&logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black?style=for-the-badge&logo=flask)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 📌 Overview

StyleForge AI is a deep learning based Neural Style Transfer application that blends the **content of one image** with the **artistic style of another** using the **AdaIN (Adaptive Instance Normalization)** algorithm.

The project includes:

- 🧠 Custom AdaIN implementation in PyTorch
- 🎨 Style transfer using a trained Decoder and pretrained VGG-19 Encoder
- 🌐 Beautiful Flask Web Interface
- ⚡ Adjustable Style Strength
- 📥 Download generated artwork instantly

---

## ✨ Features

- Upload your own content image
- Upload any artistic style image
- Adjustable style intensity (Alpha)
- Real-time style transfer
- Download stylized result
- Clean futuristic UI
- Supports JPG, JPEG and PNG images

---

## 🛠 Tech Stack

| Category | Technologies |
|----------|--------------|
| Backend | Flask |
| Deep Learning | PyTorch |
| Model | AdaIN + VGG19 Encoder + Custom Decoder |
| Frontend | HTML, CSS, JavaScript |
| Image Processing | Pillow, TorchVision |
| Deployment | Gunicorn |

---

## 🧠 Model Architecture

```
Content Image
        │
        ▼
   VGG19 Encoder
        │
        ▼
Content Features

                     AdaIN
Content Features ───────────── Style Features
        │
        ▼
 Modified Features
        │
        ▼
 Custom Decoder
        │
        ▼
 Stylized Image
```

---

## 📂 Project Structure

```
StyleForge-AI/
│
├── app.py
├── requirements.txt
├── Procfile
├── vgg_normalised.pth
│
├── experiments/
│
├── static/
│   ├── css/
│   ├── js/
│   └── uploads/
│
├── templates/
│   └── index.html
│
├── utils/
│   ├── models.py
│   └── utils.py
│
└── README.md
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/StyleForge-AI.git
```

Move into project

```bash
cd StyleForge-AI
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

Open

```
http://localhost:5000
```

---

## 📷 Demo

### Home Page

> Add your screenshot here

```
screenshots/home.png
```

---

### Generated Artwork

> Add your screenshot here

```
screenshots/result.png
```

---

## 🎯 How It Works

1. Upload a content image.
2. Upload a style image.
3. Adjust style strength.
4. Click **Upload and Style Transfer**.
5. Neural network extracts content and style features.
6. AdaIN combines them.
7. Decoder reconstructs the final artistic image.
8. Download your stylized artwork.

---

## 📈 Results

The model successfully transfers styles such as:

- Pencil Sketch
- Oil Painting
- Cubism
- Van Gogh
- Landscape Art
- Watercolor
- Abstract Painting

while preserving the original content structure.

---

## 📦 Requirements

- Python 3.10+
- Flask
- PyTorch
- TorchVision
- Pillow
- NumPy
- WTForms
- Flask-WTF
- Gunicorn

---

## 🔮 Future Improvements

- Multiple style blending
- Video Style Transfer
- Batch image processing
- Mobile responsive UI
- Image history
- GPU optimized inference
- User authentication

---

## 👨‍💻 Author

**Sorai**

AI Engineer | Machine Learning Enthusiast | Full Stack Developer

GitHub: https://github.com/yourusername

LinkedIn: https://linkedin.com/in/yourprofile

---

## ⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub!

It motivates me to build more AI-powered applications.

---

## 📜 License

This project is licensed under the MIT License.