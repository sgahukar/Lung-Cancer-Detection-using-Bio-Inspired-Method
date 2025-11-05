# 🫁 Lung Cancer Detection System using CNN and PSO

## 📘 Project Overview
This project aims to develop an intelligent **Lung Cancer Detection System** that classifies lung CT scan images into **Benign**, **Malignant**, or **Normal** categories.  
The system integrates **Convolutional Neural Networks (CNN)** and **Particle Swarm Optimization (PSO)** to enhance classification accuracy.

The web-based application allows users to:
- Upload a CT scan image.
- Choose between a **standard CNN model** or a **PSO-optimized CNN model**.
- Get the **predicted class** and **confidence score**.
- Receive **health feedback and precautions** based on the prediction.

---

## 🧠 Technologies Used

| Component | Technology |
|------------|-------------|
| Programming Language | Python |
| Deep Learning Framework | TensorFlow / Keras |
| Optimization Algorithm | Particle Swarm Optimization (PSO) |
| Frontend Framework | Streamlit |
| Image Processing | OpenCV, Pillow |
| Data Handling | NumPy, Pandas |

---


## 🧩 Modules Description

### 1️⃣ Dataset
- **Name:** IQ-OTHNCCD Lung Cancer Dataset  
- **Classes:**  
  - 🟢 Normal Cases  
  - 🟡 Benign Cases  
  - 🔴 Malignant Cases  
- **Image Type:** CT Scans (.jpg / .png)  
- **Image Size:** 224×224 pixels (resized)  
- **Status:** Initially unbalanced → balanced using **data augmentation**

---

### 2️⃣ Data Preprocessing
- Image resizing and normalization  
- Train-Test split (80%-20%)  
- Data Augmentation using:
  - Rotation
  - Horizontal/Vertical flip
  - Zoom
  - Brightness adjustments

---

### 3️⃣ CNN Models Used

| Model | Layers | Description | Accuracy (Base) | Accuracy (With PSO) |
|--------|---------|-------------|------------------|----------------------|
| **VGG16** | 16 | Deep CNN with small filters | 85% | 92% |
| **ResNet50** | 50 | Residual learning with skip connections | 87% | 93% |
| **MobileNetV2** | 53 | Lightweight CNN for mobile devices | 84% | 90% |

---

### 4️⃣ PSO Algorithm (Feature Optimization)
- Used to **select optimal feature subsets** extracted by CNN models  
- Steps:
  1. Initialize a swarm with random weights  
  2. Evaluate each particle (model accuracy = fitness)  
  3. Update velocity and position based on best scores  
  4. Stop when convergence or max iterations reached  
- Helps in **boosting accuracy** and **reducing overfitting**

---

### 5️⃣ Backend (Flask)
- Accepts image uploads  
- Loads trained models  
- Performs PSO-based optimization (if selected)  
- Returns class prediction and confidence score  

---

### 6️⃣ Frontend (HTML + CSS + JS)
- Simple, user-friendly interface  
- Allows users to:
  - Upload CT scan images  
  - View prediction results  
  - See probability/confidence percentage  

---

## ⚙️ Setup and Execution

### 🔹 Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/Lung_Cancer_Detection.git
cd Lung_Cancer_Detection
```

### 🔹 Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### 🔹 Step 3: Run Flask Server
```bash
cd backend
python app.py
```

### 🔹 Step 4: Launch Frontend
```bash
- Open frontend/index.html in a browser
- Upload an image
- View prediction and confidence score
```

## 🧰 Technologies Used

* Python 3.x
* TensorFlow / Keras
* PySwarms (for PSO optimization)
* NumPy / Pandas / OpenCV
* Matplotlib / Seaborn
* Flask
* VS Code

## 🚀 Future Enhancements
* Integrate Grad-CAM visualization to show *why* the model made its prediction.
* Add a voice feedback system for results.
* Deploy the application on Render, AWS, or Hugging Face Spaces.
* Compare PSO performance with other meta-heuristic algorithms like Genetic Algorithm (GA).

## 👩‍💻 Authors
#### Suhani Gahukar, Sakshi Bhoyar, Sukalp Warhekar
#### Branch: B.Tech in Computer Science & Engineering (AI & ML)
#### Year: 2025

