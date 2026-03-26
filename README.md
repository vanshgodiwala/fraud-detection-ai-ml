## &#x20;Fraud Detection using AI \& Machine Learning

Project Overview
This project is a simple fraud detection system built using machine learning. It predicts whether a transaction is fraudulent or not based on given features.

The system also includes a basic rule-based check along with a machine learning model, making it a combination of AI and ML concepts.



Objective
The goal of this project is to:

* Understand how classification works in machine learning
* Apply supervised learning to a real-world problem
* Learn how to evaluate models using proper metrics



Concepts Used

* Supervised Learning
* Binary Classification
* Logistic Regression
* Data Preprocessing (Feature Scaling)
* Confusion Matrix
* Precision, Recall, F1-score
* Rule-based AI system



How It Works

1. The dataset is loaded and split into training and testing data
2. The data is scaled using StandardScaler
3. A Logistic Regression model is trained on the dataset
4. A rule-based system checks for high-value transactions
5. The model predicts whether a transaction is fraud or not
6. The performance is evaluated using different metrics



Results

The model achieved:

* High accuracy (\~97%)
* High recall for fraud detection (\~0.92)

This means the model is able to detect most fraudulent transactions.



Important Insight

The dataset is highly imbalanced (very few fraud cases compared to normal transactions).  
So accuracy alone is not reliable.

Recall is more important in this case because:

> Missing a fraud is worse than incorrectly flagging a normal transaction.



Output
The program displays:

* Prediction (Fraud / Not Fraud)
* Fraud probability
* Confusion matrix
* Classification report



Technologies Used

* Python
* Pandas
* Scikit-learn
* Matplotlib



Dataset
The project uses a credit card fraud detection dataset containing transaction details and labels.



Future Improvements

* Use advanced models like Random Forest or XGBoost
* Build a real-time fraud detection system
* Improve precision while maintaining high recall



### **NOTE**

**The data set used for this project "Credit Card Fraud" is too large and hence wasn't included in the GitHub repository.**

**It ca be downloaded from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud**







Conclusion
This project demonstrates how machine learning can be used to solve real-world problems like fraud detection. It also highlights the importance of choosing the right evaluation metrics for imbalanced datasets.



### 

