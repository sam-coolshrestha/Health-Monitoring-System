
# 🩺 Lifestyle Disease Risk Prediction & Health Insights Dashboard

## 🔴 Live Dashboard
👉 [Click here to view the live dashboard](https://health-monitoring-system-s6wr65wid.vercel.app)

---

## 📌 Project Overview
This project analyzes health-related data to predict the **risk of lifestyle diseases** based on an individual's health parameters. Users input basic health information such as **blood pressure, heart rate, smoking status, BMI, and other lifestyle indicators**, and the system analyzes this data to estimate potential disease risks.

The project also generates **visual insights through a dashboard** to help users better understand their health patterns and risk factors.

The goal of this project is to demonstrate how **data analysis and predictive modeling can assist in early detection of lifestyle diseases and promote preventive healthcare.**

---

## 🎯 Objectives
* Analyze healthcare datasets to identify patterns related to lifestyle diseases.
* Predict disease risk based on user-provided health parameters.
* Provide **data-driven health insights** through visualization dashboards.
* Help users understand how their lifestyle factors affect their health.

---

## 🧠 Key Features
✔ Health data preprocessing and cleaning
✔ Lifestyle disease risk prediction using data analysis techniques
✔ User input system for personal health parameters
✔ Risk evaluation based on dataset insights
✔ Interactive health insights dashboard
✔ Data visualization for better understanding of health patterns

---

## 🤖 ML Model Results
The backend trains and compares **6 machine learning models** on the NHANES dataset (5,735 participants):

| Model | Accuracy | F1 Score | AUC-ROC |
|---|---|---|---|
| Logistic Regression | 95.12% | 0.9712 | 0.9777 |
| Decision Tree | 99.91% | 0.9995 | 0.9973 |
| Random Forest | 99.74% | 0.9984 | 1.0000 |
| **Gradient Boosting** ✅ | **99.91%** | **0.9995** | **1.0000** |
| SVM | 97.04% | 0.9825 | 0.9943 |
| KNN | 96.34% | 0.9783 | 0.9881 |

**Best Model: Gradient Boosting (AUC = 1.0000)**

### Training Details
- **Dataset:** NHANES (National Health and Nutrition Examination Survey)
- **Samples:** 5,735 participants
- **Features:** 17 (including 5 engineered features)
- **Train/Test Split:** 80% / 20% stratified
- **Cross-Validation:** 5-Fold Stratified KFold

---

## 📊 Input Parameters
The system analyzes the following user inputs:
* Blood Pressure (Systolic & Diastolic)
* Heart Rate
* BMI
* Smoking Status
* Age
* Waist Circumference
* Physical Activity Level
* Other lifestyle-related indicators (based on dataset features)

Using these parameters, the system estimates the **risk probability of lifestyle-related diseases.**

---

## 📈 Output
The system provides:
* Disease risk estimation
* Health insights based on dataset trends
* Visual dashboard displaying:
  * Health indicators
  * Risk distribution
  * Lifestyle impact on disease probability

---

## 🛠 Technologies Used

### Backend / ML
* **Python 3**
* **Pandas** – Data preprocessing and analysis
* **NumPy** – Numerical operations
* **Matplotlib / Seaborn** – Data visualization
* **Scikit-learn** – Machine learning models
* **SciPy** – Statistical analysis

### Frontend / Dashboard
* **React** – Frontend framework
* **Vite** – Build tool
* **Recharts** – Interactive charts
* **Vercel** – Deployment

---

## 🚀 Run the ML Backend Locally

### 1. Clone the repo
```bash
git clone https://github.com/sam-coolshrestha/Health-Monitoring-System.git
cd Health-Monitoring-System
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### 3. Run the pipeline
```bash
python lifestyle_risk_model.py
```

This will print all model results and generate 4 plots in an `outputs/` folder.

---

## 🖥 Run the Dashboard Locally

```bash
cd dashboard
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 📂 Project Workflow
1️⃣ Data Collection
2️⃣ Data Preprocessing
3️⃣ Exploratory Data Analysis (EDA)
4️⃣ Feature Engineering
5️⃣ Model Training & Evaluation
6️⃣ Dashboard Visualization
7️⃣ User Health Input System (Risk Calculator)

---

## 📊 Dataset Analysis
The NHANES dataset contains multiple health and lifestyle parameters analyzed to identify correlations between **lifestyle habits and disease risks.**

Key findings:
* **59.4%** of participants are smokers — largest single risk factor
* **71.2%** are overweight or obese (BMI ≥ 25)
* Hypertension rate jumps from **13%** (age 18–30) to **59%** (age 60+)
* BMI and waist circumference are nearly perfectly correlated **(r = 0.91)**

---

## 🔮 Future Scope

### 1️⃣ Computer Vision Based Rehabilitation Assistant
A future version aims to integrate **OpenCV-based rehabilitation exercise monitoring**, where:
* Patients perform rehabilitation exercises
* The system detects body posture using computer vision
* It provides feedback on whether exercises are performed correctly

This feature is **currently proposed as a future enhancement and has not yet been implemented.**

### 2️⃣ Machine Learning Model Improvements
* Train deep learning models (Neural Networks)
* Improve accuracy of disease risk prediction

### 3️⃣ Real-Time Health Monitoring
Integration with wearable devices for real-time health data.

### 4️⃣ Backend API
Build a REST API so the dashboard can call the ML model directly in real time.

---

## 📜 License
This project is open source and available under the **MIT License**.


