# Module 12 Report Template

## Overview of the Analysis

* Purpose of the Analysis
    Credit Risk is the risk that the bank/lender may not recoup all or some of the money lent to the borrower due to a borrower default. Lenders consume various data to determine the credit risk associated with lending to any particular borrower. In this analysis, we used a machine learning model to determine which loans are healthy (low-risk), and which loans are not (high-risk). Higer risk loans are typically associated with higher levels of default.
    The purpose of the analysis was to create a logistic Regression model, suppply training data to the model and have have it make predictions based on the training data. The relevant data for the model is loan data containing various columns including debt to income, income, debt, loan, interest, derogatory marks and loan_status. 

* Type of Financial information and what needed to be predicted
    The financial data included loan size, debt to income, borrower income, total debt, loan, interest, derogatory marks and loan_status. The model needed to predict wheather the loan was healthy (0) or high risk (1) based on the financial data. 

* Basic information about the variables we were trying to predict (e.g., `value_counts`).
    The dataset included 77,536 records with a maximum loan size of $23,000 and a minimum of $5,000, with a mean of $9,805. There were other details related to the financial information listed above for all of the 77 thousand records. THe machine learning model ingested all these data points to provide a view of which of the loans were healthy, and which were not. 

* Describe the stages of the machine learning process you went through as part of this analysis.

    After importing the data, I separated the features (variables) from the labels (the risk associated with each borrower). The next step was to use a train_test_learn module to do a random split of the data into training vs testing. Separating and splitting the data is important because it allows the model to learn and make predictions without having seen the results of the features. In other words, a model that makes predictions after having seen the results would spit out almost perfect results, leading to overfitting. To avoide this, we hold out part of the available data as a test set (X_test, y_test).

    The next step is to fit a logistic regression model with the original data. This consisted of usign a LogisticRegression function of sklearn.linear_modoel module, which has inbuilt algorythm (lbfgs) to perform the regression. Fitting the model to our train data 'trained' the model on the a portion of the starting data and outcomes. Essentially, logistic regression is a classifier that accomplishes binary classification by predicting the probability of an outcome, even or observation. This approach is perfect for our purposes becase we are interested in whether a loan is healthy (0) ,or not (1). 

    I created a predictions array, which held the predictions made by the classifier/logistic regression model on test data. This was compared with the actual data and subsequent analysis performed to determine the model's efficacy. 


## Results

* Machine Learning Model 1:
        Class 0 (Healthy Loans):

            Precision: 1
            Recall: 1
            F1-Score: 1

        Class 1 (High-risk Loans):

            Precision: 0.87
            Recall: 0.89
            F1-Score: 1

        Overall Accuracy: 0.99

        Macro Average:

            Precision: 0.94
            Recall: 0.94
            F1-Score: 0.94

        Weighted Average:

            Precision: 0.99
            Recall: 0.99
            F1-Score: 0.99
    

## Summary and Recommendations

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:

Based on a review of the confusion matrix, the model appears to have predicted the healthy loans (18679) correctly as low risk (True Negative), while 558 loans were correctely predicted as high risk (True Positive). Accordingly, 67 high risk loans were incorrectly predicted as low risk (False Negative), while 80 healthly loans were incorrectly predicted as high risk (False Positive).

Moving on to the classification report, the precision, recall, and f1-score of Healthy loans are all 100%, indicating very reliable, well performing model overall. However, additionally analysis would need to be performed to determine whether there are instances of overfitting, particularly for the healthy loans. The precision, recall, and f1 score of the high-risk loans are 87%, 89%, and 88% respectively. The model still performs well although a sligtly lower precision and recall metrics. This might indicate that there is room for improvement in distinguishing high-risk loans from healthy ones.

The overall accuracy of 99% and macro and weighted averages of 94% and 99% across the metrics indicate a very dependable model and I would recommend it for use. 
