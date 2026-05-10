# ML-Based Calories Burn Prediction System

This project is a machine learning-based fitness tracking application developed using Python, Streamlit, and scikit-learn. The application predicts the number of calories burned during exercise based on user inputs such as age, BMI, exercise duration, heart rate, body temperature, and gender.

The main objective of this project was to build an interactive and user-friendly fitness prediction system while learning how machine learning models can be integrated into real-world web applications.

The project uses Random Forest Regression as the primary prediction model and also includes Linear Regression for comparison purposes. A modern Streamlit interface was designed to make the application simple, responsive, and easy to use.

---

# Features

* Real-time Calories Burn Prediction
* Machine Learning-based prediction system
* Random Forest Regression & Linear Regression models
* BMI auto-calculation using height and weight
* Modern Streamlit dashboard UI
* Interactive form-based user input system
* Responsive web application
* Custom loading animation during prediction
* Health analytics and fitness tracking support
* Built completely using Python

---

# Tech Stack

| Technology               | Usage                     |
| ------------------------ | ------------------------- |
| Python                   | Core Programming Language |
| Streamlit                | Web Application Framework |
| scikit-learn             | Machine Learning Models   |
| Pandas                   | Data Processing           |
| NumPy                    | Numerical Computation     |
| Random Forest Regression | Prediction Model          |
| Linear Regression        | Comparison Model          |

---

# Project Structure

```bash
ML-Based-Calories-Burn-Prediction-System/
│
├── app.py
├── calories.csv
├── exercise.csv
├── requirements.txt
├── README.md
```

---

# Machine Learning Workflow

## 1. Data Collection

The project uses two datasets:

* `exercise.csv`
* `calories.csv`

These datasets contain:

* Age
* Gender
* Height
* Weight
* Heart Rate
* Body Temperature
* Exercise Duration
* Calories Burned

---

## 2. Data Preprocessing

The preprocessing pipeline includes:

* Merging datasets using `User_ID`
* Removing duplicate records
* Encoding gender values
* Creating BMI feature
* Selecting important features for prediction

---

## 3. Feature Engineering

BMI is calculated dynamically using:

```python
BMI = Weight / ((Height / 100) ** 2)
```

Features used for prediction:

* Age
* BMI
* Duration
* Heart Rate
* Body Temperature
* Gender

---

## 4. Model Training

Two Machine Learning models are trained:

### Linear Regression

Used as a baseline regression model.

### Random Forest Regression

Used as the primary prediction model due to better prediction accuracy.

---

# Prediction Process

1. User enters fitness and health details.
2. BMI is calculated automatically.
3. Data is passed to the trained ML model.
4. Random Forest model predicts calories burned.
5. Prediction is displayed instantly on the dashboard.

---

# User Interface

The application includes:

* Interactive input form
* Responsive dashboard design
* Real-time prediction system
* Modern dark-themed UI
* Animated loading overlay
* Fitness analytics display

---

# Installation Guide

## 1. Clone Repository

```bash
git clone https://github.com/sreejannn/ML-Based-Calories-Burn-Prediction-System.git
```

---

## 2. Navigate to Project Folder

```bash
cd ML-Based-Calories-Burn-Prediction-System
```

---

## 3. Create Virtual Environment (Optional)

### Windows

```bash
python -m venv venv
```

Activate Environment:

```bash
venv\Scripts\activate
```

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Run the Application

```bash
python -m streamlit run app.py
```

---

# Requirements

```text
streamlit
pandas
numpy
scikit-learn
```

---

# Project Preview

## Dashboard Features

* User Input Form
* Calories Burn Prediction
* BMI Calculation
* Machine Learning Prediction Engine
* Interactive UI Components

---

# Future Enhancements

* Mobile Responsive Design
* Cloud Deployment
* Deep Learning Integration
* Advanced Fitness Analytics
* Workout Recommendation System
* Diet Recommendation Module
* Real-time Graph Visualizations

---

# Learning Outcomes

This project helped in understanding:

* Machine Learning workflow
* Data preprocessing techniques
* Feature engineering
* Regression models
* Streamlit web development
* Model deployment concepts
* UI/UX integration with ML systems

---

# Author

### Sreejan Narapareddy

Computer Science Engineering (Cybersecurity)

Interested in Machine Learning, Cybersecurity, and Software Development.
