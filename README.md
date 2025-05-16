# 💓 Heart Failure Prediction using Machine Learning

This project aims to predict heart failure based on clinical records using various machine learning algorithms. By analyzing patient health metrics, the models can assist in identifying individuals at risk of heart failure.

## 📁 Project Structure

```
heart-failure/
├── dataset/
│   └── heart_failure_clinical_records_dataset.csv
├── death/
├── disease/
├── output/
├── library.py
├── main.py
├── requirements.txt
└── README.md
```

## ⚙️ How to Run

1. **Clone the Repository**

```bash
git clone https://github.com/mahdijafari78/heart-failure.git
cd heart-failure
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the Main Script**

```bash
python main.py
```

> Make sure you have Python 3.7+ installed and a working virtual environment (recommended).

## 🧠 Models Used

The following machine learning models were implemented and evaluated:

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- XGBoost

## 📊 Evaluation Metrics

Models are evaluated using standard metrics:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**

## 📈 Dataset

The dataset used in this project is the [Heart Failure Clinical Records Dataset](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data), which contains 13 clinical features collected from 299 patients.

## ✅ Features

- Cleaned and preprocessed dataset.
- Model comparison and performance visualization.
- Modular code structure for better readability and reusability.
- Easy to extend with new models or features.

## 📌 TODO (Optional Enhancements)

- Add hyperparameter tuning (GridSearchCV, Optuna, etc.)
- Implement cross-validation strategies
- Deploy as a web application (e.g., using Flask or Streamlit)


---

Feel free to fork, modify, and contribute!

Made with ❤️ by [@mahdijafari78](https://github.com/mahdijafari78)
