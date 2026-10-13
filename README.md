# Machine Learning Projects (CodSoft Internship)

These are the three machine learning projects I made during my one-month Machine Learning internship at **CodSoft** in July 2025. All the notebooks are written in Python and were run on Kaggle.

## Projects

1. [Customer Churn Prediction](#1-customer-churn-prediction)
2. [Credit Card Fraud Detection](#2-credit-card-fraud-detection)
3. [Movie Genre Classification](#3-movie-genre-classification)

---

## 1. Customer Churn Prediction

The goal is to predict whether a bank customer will leave the bank (churn) or stay.

**Dataset:** `Churn_Modelling.csv` with 10,000 customers and 14 columns. About 20% of the customers churned (2,037 out of 10,000).

**What I did:**
- Checked the data types and missing values (there were no missing values)
- Removed columns that are not useful for prediction (`RowNumber`, `CustomerId`, `Surname`)
- One-hot encoded `Geography` and `Gender`
- Plotted the churn count to see the class balance
- Scaled the features with `StandardScaler`
- Split the data into 90% training and 10% testing
- Trained and compared 6 models

**Results:**

| Model | Accuracy |
|---|---|
| Gradient Boosting | **86.7%** |
| SVM | 86.5% |
| Random Forest | 86.5% |
| KNN | 84.0% |
| Logistic Regression | 80.9% |
| Decision Tree | 80.0% |

Gradient Boosting gave the best accuracy of **86.7%**.

---

## 2. Credit Card Fraud Detection

The goal is to find fraud transactions in credit card data.

**Dataset:** `fraudTrain.csv` (1,296,675 rows) and `fraudTest.csv` (555,719 rows), which makes 1,852,394 transactions in total. Only about **0.5%** of them are fraud, so the data is highly imbalanced.

**What I did:**
- Combined the train and test files and checked for missing values
- Label encoded `merchant`, `category`, `gender`, `state` and `job`
- Created new features from the date columns: transaction year, month, day and hour, and birth year, month and day
- Removed personal details that are not useful (`first`, `last`, `street`, `city`, `trans_num`)
- EDA: fraud vs non-fraud count, transactions by category, transactions by gender and a correlation heatmap
- Split the data into 80% training and 20% testing
- Trained Logistic Regression and Decision Tree models

**Results (on 370,479 test transactions):**

| Model | Accuracy | Fraud Precision | Fraud Recall | Fraud F1-score |
|---|---|---|---|---|
| Logistic Regression | 99.47% | 0.00 | 0.00 | 0.00 |
| Decision Tree | **99.81%** | **0.80** | **0.84** | **0.82** |

Logistic Regression predicted every transaction as "not fraud", so its accuracy looks high only because fraud is so rare. The **Decision Tree** actually caught **84%** of the fraud cases with an F1-score of **0.82**, so it is the better model here.

---

## 3. Movie Genre Classification

The goal is to predict the genre of a movie from its plot description.

**Dataset:** 54,214 movies with title, genre and description, across **27 genres**. A separate test file has 54,200 movies.

**What I did:**
- Plotted the number of movies in each genre
- Cleaned the text: lowercase, removed URLs, handles, punctuation and very short words, tokenized with NLTK and removed English stopwords
- Plotted the length of the cleaned descriptions
- Converted the text to numbers with **TF-IDF**
- Split the training data into 80% training and 20% validation
- Trained a **Multinomial Naive Bayes** classifier
- Predicted genres for the test movies and saved them to `predicted_genres.csv`

**Result:** 44.5% validation accuracy across 27 genres. The model does best on the biggest genres like drama and documentary.

---

## Tools Used

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, NLTK

## How to Run

1. Clone this repository.
2. Install the libraries:
```
   pip install pandas numpy matplotlib seaborn scikit-learn nltk
```
3. Download the datasets from Kaggle and change the file paths in the notebooks (they currently use `/kaggle/input/...`).
4. For the movie genre project, download the NLTK data once:
```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
```
5. Open the notebooks in Jupyter Notebook or upload them to Kaggle and run all cells.

## Kaggle Notebooks

- [Customer Churn Prediction](https://www.kaggle.com/code/vishalbariar/customer-churn-prediction)
- [Credit Card Fraud Detection](https://www.kaggle.com/code/vishalbariar/credit-card-fraud-detection)
- [Movie Genre Classification](https://www.kaggle.com/code/vishalbariar/movie-genre-classification-task)

## Author

**Vishal Bariar**

B.Tech, Electronics and Communication Engineering, NIET Greater Noida

[LinkedIn](https://www.linkedin.com/in/vishal-bariar-431772349) | [GitHub](https://github.com/Vishalbariar)
