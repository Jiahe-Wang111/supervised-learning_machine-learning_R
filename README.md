# supervised-learning_machine-learning_R
# Machine Learning Projects (Supervised Learning)

This repository contains projects completed as part of my coursework in supervised machine learning.  
Each project explores different models and approaches, with a focus on understanding **prediction vs. inference**, **bias-variance tradeoff**, and the challenges of working with high- vs. low-dimensional data.  

---

## 📂 Project Structure

### Part 1: Predicting Tweet Authors (Bernie Sanders vs. Donald Trump)
- **Objective:** Predict the author of a tweet (Trump or Bernie) based on its word content.  
- **Models:** Logistic Regression, Ridge Regression  
- **Key Steps:**  
  - Data import and exploration (document-term matrix)  
  - Logistic regression (training accuracy and cross-validation)  
  - Ridge regression with cross-validated λ (regularization parameter)  
  - Bias-variance interpretation  
- **Takeaway:** Logistic regression struggles in high-dimensional settings → regularization (ridge) improves predictive performance.

---

### Part 2: Social Network Ad Purchase Prediction
- **Objective:** Predict whether a user will purchase a product based on **Age, Gender, and Salary**.  
- **Models:** Logistic Regression, GAM (Generalized Additive Models)  
- **Key Steps:**  
  - Logistic regression with cross-validation  
  - GAMs with different spline degrees of freedom (`ns()` function for Age & Salary)  
  - Comparison of model performance (bias vs. variance)  
  - Visualization of non-linear relationships using `ggeffects::ggpredict()`  
- **Takeaway:** GAMs capture non-linear effects and can outperform logistic regression when linearity assumptions are too restrictive.

---

### Part 3: K-Fold Cross-Validation From Scratch (Bonus)
- **Objective:** Implement a k-fold cross-validation function manually to select the best polynomial degree for a regression model.  
- **Models:** Polynomial regression  
- **Key Steps:**  
  - Write custom CV function (`cv_poly`)  
  - Compare MSE across degrees (linear, quadratic, cubic, etc.)  
- **Takeaway:** This part is more about learning how CV works internally (since in practice we use packages like `caret`).  

---

## 🚀 Key Learning Points
- Logistic regression is simple but may fail in **high-dimensional** or **non-linear** settings.  
- **Regularization (ridge/lasso)** helps control overfitting.  
- **GAMs** extend flexibility by modeling non-linear relationships.  
- **Cross-validation (CV)** is a universal tool to evaluate models and prevent over/underfitting.  

---

## 📌 How to Run
1. Clone this repository:  
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
