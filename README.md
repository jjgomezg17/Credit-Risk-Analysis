# Credit Risk Analysis

## Overview of the analysis:

### The purpose of this analysis is to use machine learning techniques to build and evaluate models for predicting credit risk using an imbalanced dataset from LendingClub, a peer-to-peer lending services company. The dataset contains information about the credit history of borrowers, as well as other relevant features such as loan amount, interest rate, and employment length.

### Since credit risk is an inherently unbalanced classification problem, with good loans easily outnumbering risky loans, we need to employ different techniques to train and evaluate models with unbalanced classes. Therefore, we will use various resampling techniques such as oversampling, undersampling, and combination of both to balance the dataset.

### We will use the imbalanced-learn and scikit-learn libraries to build and evaluate different machine learning models, including the BalancedRandomForestClassifier and EasyEnsembleClassifier, which are designed to reduce bias in the predictions. We will compare the performance of these models and evaluate their suitability for predicting credit risk.

### The ultimate goal of this analysis is to provide a recommendation on whether these machine learning models should be used by LendingClub to predict credit risk and make informed lending decisions.

## Results

### Naive Random Oversampling

#### The balanced accuracy score and precision and recall scores for Naive Random Oversampling can be summarized as follows:

#### Balanced accuracy score:

#### The balanced accuracy score for Naive Random Oversampling is 0.583.

#### Precision and Recall scores:

#### The precision score for predicting high risk loans is very low at 0.01, meaning that out of all the loans predicted as high risk, only 1% were actually high risk.
#### The recall score for predicting high risk loans is moderate at 0.52, meaning that out of all the actual high risk loans, only 52% were correctly identified as high risk.
#### The precision score for predicting low risk loans is very high at 1.0, meaning that out of all the loans predicted as low risk, 100% were actually low risk.
#### The recall score for predicting low risk loans is moderate at 0.64, meaning that out of all the actual low risk loans, 64% were correctly identified as low risk.

#### Overall, the precision and recall scores for predicting high risk loans are low, indicating that this model may not be suitable for predicting credit risk and may require further improvement.

### SMOTE Oversampling

#### The balanced accuracy score and precision and recall scores for SMOTE Oversampling can be summarized as follows:

#### Balanced accuracy score:

#### The balanced accuracy score for SMOTE Oversampling is 0.617.

#### Precision and Recall scores:

#### The precision score for predicting high risk loans is very low at 0.01, meaning that out of all the loans predicted as high risk, only 1% were actually high risk. This is similar to the previous model with Naive Random Oversampling.
#### The recall score for predicting high risk loans is moderate at 0.51, meaning that out of all the actual high risk loans, only 51% were correctly identified as high risk. This is lower than the previous model with Naive Random Oversampling.
#### The precision score for predicting low risk loans is very high at 1.0, meaning that out of all the loans predicted as low risk, 100% were actually low risk.
#### The recall score for predicting low risk loans is moderate at 0.72, meaning that out of all the actual low risk loans, 72% were correctly identified as low risk.

#### Overall, the precision and recall scores for predicting high risk loans are still low, indicating that this model may still not be suitable for predicting credit risk and may require further improvement. However, the slightly higher balanced accuracy score may suggest that SMOTE Oversampling has improved the model's performance to some extent.

### ClusterCentroids resampler

#### Balanced accuracy score:

#### The balanced accuracy score obtained for this model was 0.537, which is significantly lower than the previous two models using Naive Random Oversampling and SMOTE oversampling techniques.

#### This suggests that undersampling using ClusterCentroids may not be the best technique for this dataset and may have decreased the performance of the model in predicting credit risk.

#### Precision and recall scores:

#### The imbalanced classification report shows that the model has very low precision and specificity for predicting high risk loans, with a precision score of only 0.01 and a recall score of 0.52.
#### This means that out of all the loans predicted as high risk, only 1% are actually high risk, and the model is missing nearly half of all high risk loans.
#### The recall score for low risk loans is also low, at 0.55, which means that the model is correctly identifying only about half of all low risk loans.
#### The F1 score for high risk loans is also very low at only 0.01, indicating that the model is not performing well in identifying high risk loans.

####Overall, these precision and recall scores suggest that the model trained using ClusterCentroids is not suitable for predicting credit risk and may require further improvement using different resampling techniques or machine learning models.

#### Confusion matrix:

#### The confusion matrix shows that the model correctly predicted 46 true positive and 7534 true negative cases.
#### However, the model also predicted 42 false positive and 6142 false negative cases.
#### This means that the model is still not performing well in identifying high risk loans, and is also misclassifying a significant number of low risk loans as high risk.

### SMOTEENN

#### Balanced accuracy score is 0.658, indicating better performance than Random Oversampling and ClusterCentroids resampling methods.

### Precision for high-risk loans is low at 0.01, meaning that only 1% of predicted high-risk loans are actually high-risk.

#### Recall for high-risk loans is high at 0.69, indicating that 69% of true high-risk loans are correctly classified as high-risk.

#### Precision for low-risk loans is perfect at 1.0, meaning that all predicted low-risk loans are actually low-risk.

#### Recall for low-risk loans is 0.62, indicating that 62% of true low-risk loans are correctly classified as low-risk.

#### The F1 score is 0.76, indicating reasonable overall performance.

#### The geometric mean is 0.66, indicating that the model is performing better than random guessing but still has room for improvement.

#### The classification report demonstrates that the SMOTEENN resampling method performs better than the previous three resampling methods, with higher recall for high-risk loans and better precision and recall for low-risk loans.

### Balanced Random Forest Classifier

#### The balanced accuracy score for the Balanced Random Forest Classifier is 0.6417, which is better than the score of the Naive Random Oversampling and ClusterCentroids resampling techniques but slightly lower than the SMOTE Oversampling and SMOTEENN algorithms.

#### The precision score for the high-risk class is 0.78, which means that out of all the predicted high-risk loans, 78% of them are true high-risk loans. The recall score for the high-risk class is 0.28, which means that out of all the actual high-risk loans, only 28% of them were correctly identified as high-risk loans.

#### The precision and recall scores for the low-risk class are both 1.00, which means that out of all the predicted low-risk loans, 100% of them are true low-risk loans, and out of all the actual low-risk loans, 100% of them were correctly identified as low-risk loans.

#### The feature importances are listed in descending order, with total_rec_prncp, total_rec_int, and total_pymnt being the top three features.

### Easy Ensemble AdaBoost Classifier

#### Balanced accuracy score: 0.9069
#### Precision and recall scores:

#### For the high risk class:
#### Precision: 0.17
#### Recall: 0.84

#### For the low risk class:
#### Precision: 1.00
#### Recall: 0.97

#### These results suggest that the Easy Ensemble AdaBoost Classifier performs significantly better than the other models we have evaluated so far. It achieves a high balanced accuracy score of 0.9069, which is much better than the other models. The precision and recall scores for the high risk class are still low, but they are much better than what we have seen with the other models. The precision for the high risk class is 0.17, which means that only 17% of the loans that we classify as high risk are actually high risk. However, the recall for the high risk class is 0.84, which means that we are able to correctly identify 84% of the actual high risk loans. The precision and recall scores for the low risk class are both very high, which means that we are able to correctly identify the vast majority of the low risk loans.

## Summary

#### After evaluating five different machine learning models, here are the key results:

#### Naive Random Oversampling:
#### Balanced Accuracy: 0.5831
#### Precision: high_risk (0.01), low_risk (1.00)
#### Recall: high_risk (0.52), low_risk (0.64)

#### SMOTE Oversampling:
#### Balanced Accuracy: 0.6166
#### Precision: high_risk (0.01), low_risk (1.00)
#### Recall: high_risk (0.51), low_risk (0.72)

#### ClusterCentroids Resampler:
#### Balanced Accuracy: 0.5368
#### Precision: high_risk (0.01), low_risk (0.99)
#### Recall: high_risk (0.52), low_risk (0.55)

#### SMOTEENN Resampler:
#### Balanced Accuracy: 0.6580
#### Precision: high_risk (0.01), low_risk (1.00)
#### Recall: high_risk (0.69), low_risk (0.62)

#### Balanced Random Forest Classifier:
#### Balanced Accuracy: 0.6418
#### Precision: high_risk (0.78), low_risk (1.00)
#### Recall: high_risk (0.28), low_risk (1.00)

#### Easy Ensemble AdaBoost Classifier:
#### Balanced Accuracy: 0.9069
#### Precision: high_risk (0.17), low_risk (1.00)
#### Recall: high_risk (0.84), low_risk (0.97)

#### Based on the above results, the Easy Ensemble AdaBoost Classifier is the best model as it has the highest balanced accuracy score (0.9069), which indicates that it performs well in predicting both the minority and majority classes. Additionally, it has high precision and recall scores for both classes, which means that it is effective at identifying and correctly classifying both high-risk and low-risk loans. Therefore, I would recommend using the Easy Ensemble AdaBoost Classifier model to classify loans as high-risk or low-risk.
