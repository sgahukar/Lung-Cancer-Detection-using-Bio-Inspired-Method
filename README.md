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

## 🧩 Project Architecture

lung_cancer_app/
│
├── app.py # Main Streamlit app (frontend)
│
├── backend/ # Core backend logic
│ ├── model_loader.py # Loads CNN or PSO-optimized model
│ ├── preprocess.py # Image resizing & normalization
│ ├── predictor.py # Model prediction & confidence output
│ └── evaluation.py # Optional: accuracy, confusion matrix, etc.
│
├── models/ # Saved trained models
│ ├── cnn_model.h5 # Basic CNN model
│ ├── cnn_pso_model.h5 # PSO-enhanced CNN model
│
├── dataset/ # Dataset folder
│ ├── lung dataset/ # Main dataset
│ │ ├── Bengin cases/
│ │ ├── Malignant cases/
│ │ └── Normal cases/
│ └── Test cases/ # External test samples
│
├── utils/ # Helper modules
│ ├── feedback_generator.py # Generates feedback & suggestions
│ └── visualization.py # (optional) Grad-CAM or charts
│
├── reports/ # Training & testing results
│ ├── model_performance.txt
│ ├── confusion_matrix.png
│ └── training_history.png
│
├── requirements.txt # All required libraries
└── README.md # Project documentation

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

### 🔹 Step 2: Install Dependencies
pip install -r requirements.txt

###🔹 Step 3: Run Flask Server
cd backend
python app.py

### 🔹 Step 4: Launch Frontend
- Open frontend/index.html in a browser
- Upload an image
- View prediction and confidence score

### 🧰 Technologies Used

- Python 3.x
- TensorFlow / Keras\
- PySwarms (for PSO optimization)
- NumPy / Pandas / OpenCV
- Matplotlib / Seaborn
- Streamlit
- VS Code

### 🚀 Future Enhancements

- Integrate Grad-CAM visualization
- Add voice feedback system
- Deploy app on Render / AWS / HuggingFace Spaces
- Compare PSO with Genetic Algorithm (GA)

👩‍💻 Author

Name: Suhani Gahukar, Sakshi Bhoyar, Sukalp Warhekar
Branch: B.Tech CSE (AI & ML)
Year: 2025