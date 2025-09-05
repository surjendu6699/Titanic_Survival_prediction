# Titanic_Survival_prediction
Machine Learning project predicting survival on the Titanic dataset using Python &amp; Scikit-learn.
#  Titanic Survival Prediction  

##  Overview  
This project predicts the survival of passengers on the Titanic using Machine Learning.  
It is built using the **Titanic dataset** from Kaggle, which includes passenger details such as age, sex, ticket class, number of siblings/spouses, and more.  

The goal is to demonstrate **data preprocessing, feature engineering, and building a prediction pipeline** using Scikit-learn.  

## Tech Stack  
- Python   
- Pandas, NumPy  
- Scikit-learn (Logistic Regression, Pipelines, Preprocessing)  
- Matplotlib / Seaborn (for data visualization)  

##  Workflow  
1. **Data Cleaning** → Handle missing values & drop irrelevant features  
2. **Feature Encoding** → Convert categorical variables (Sex, Embarked) into numeric using OneHotEncoder  
3. **Scaling** → Standardize numeric features (Age, Fare)  
4. **Model Training** → Logistic Regression with Scikit-learn pipeline  
5. **Evaluation** → Accuracy score & confusion matrix  

##  Results  
- Achieved an accuracy of **XX%** on the test set.  
- Observed that **gender and passenger class** were strong predictors of survival.  

##  Sample Visualization  
*(You can add a screenshot here, e.g., survival distribution by gender or confusion matrix plot)*  

##  How to Run  

# Clone the repository
git clone https://github.com/yourusername/titanic-survival-prediction.git  

# Install dependencies
pip install -r requirements.txt  

# Run the notebook
jupyter notebook Titanic.ipynb
