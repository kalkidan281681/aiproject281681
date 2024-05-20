## Project for AI: <br>
# SafeComm Digital Security Solutions
Authors: <br> <i> Ana Andrijasevic </i> <br> <i> Cherine Abboud </i> <br> <i> Kalkidan Mezgebe </i> <br>

## Introduction <br>
In the modern digital age, text messaging has become a primary mode of communication for people worldwide. However, this convenience comes with a significant drawback: the proliferation of SMS-based fraud. These fraudulent messages often aim to deceive recipients, leading to financial losses and other adverse outcomes. Our project, conducted, aims to design and implement a machine learning-based system that can automatically identify and flag fraudulent SMS messages. By leveraging anonymized SMS data provided by a major telecom provider, we aim to enhance digital security and protect users from potential fraud.<br>
<br>

## Methods <br>
To achieve our objective, we adopted a systematic approach encompassing data preprocessing, feature extraction, model training, and evaluation. The dataset comprised anonymized SMS messages labeled as either fraudulent or non-fraudulent. Our methodology included the following key steps:
Data Preprocessing <br>
We started by handling missing values in the dataset. Instead of dropping these records, we opted for imputation to retain as much data as possible. Next, we focused on feature extraction. We transformed the SMS text into numerical features using the TF-IDF (Term Frequency-Inverse Document Frequency) vectorization technique. This approach allowed us to capture the importance of words in each message relative to the entire dataset. Additionally, we calculated the length of each SMS as a supplementary feature.
Model Training <br>
Recognizing that our task is a binary classification problem, we experimented with three machine learning algorithms: Naive Bayes, Random Forest, and XGBoost. Each model was trained and tuned using GridSearchCV to identify the optimal hyperparameters. We divided the data into training, validation, and test sets (60%-20%-20%) to ensure robust evaluation and prevent overfitting. <br>
<br>
Environment Setup <br>
To facilitate reproducibility, our project environment was managed using Anaconda. The following commands can recreate the environment:<br>

conda create --name safecomm_env python=3.8 <br>
conda activate safecomm_env <br>
pip install pandas numpy scikit-learn matplotlib seaborn xgboost <br>
<br>
Design Choices <br>
Our choice of models was influenced by their respective strengths: Naive Bayes for its simplicity and effectiveness in text classification, Random Forest for its robustness through ensemble learning, and XGBoost for superior performance via gradient boosting. Comparing these models helped identify the most effective approach for detecting fraudulent SMS.<br>
<br>

## Experimental Design <br>
<i> Main Purpose </i> <br>
<be>
The primary objective of our experiments was to determine the most effective machine learning model for identifying fraudulent SMS messages.<br>
<i> Baseline </i> <br>
<br>
We established a baseline using the Naive Bayes classifier, a commonly used algorithm for text classification tasks.<br>
<i> Evaluation Metrics</i> <br>
We employed several evaluation metrics to assess model performance, including accuracy, precision, recall, F1 score, and ROC AUC. Precision and recall were particularly important given the cost of false positives and false negatives in fraud detection.<br>
<br>

## Results <br>
<i> Main Findings </i> <br>
After evaluating the models on the validation set, the Random Forest classifier emerged as the most effective, with an accuracy of 97.85%, precision of 100%, recall of 85.19%, and an F1 score of 92%. The ROC AUC was 98.80%, indicating excellent discriminatory ability.<br>
Figures and Tables<br>
Model Performance Comparison
<br>
Model	Accuracy	Precision	Recall	F1 Score	ROC AUC
Naive Bayes	98,21%	96,10%	91,36%	93,67%	98,43%
Random Forest	97,85%	100,00%	85,19%	92,00%	98,90%
XGBoost	97,67%	97,87%	86,42%	91,50%	98,42%<img width="692" alt="image" src="https://github.com/kalkidan281681/aiproject281681/assets/170321639/d6820523-8283-41f3-a03f-3a24ca9ffc11">
<img width="692" alt="image" src="https://github.com/kalkidan281681/aiproject281681/assets/170321639/d6820523-8283-41f3-a03f-3a24ca9ffc11">


## Conclusions<br>
Take-Away Point<br>
Our study highlights the effectiveness of machine learning models in detecting fraudulent SMS messages. The Random Forest classifier, in particular, demonstrated excellent performance with an accuracy of 97.85%, precision of 100%, recall of 85.19%, and an F1 score of 92%. These results suggest that machine learning can significantly enhance digital security by automatically identifying and flagging potentially harmful messages, thereby protecting users from fraud.<br>
Insights and Implications<br>
The high precision of the Random Forest model indicates its reliability in correctly identifying fraudulent messages, which is crucial for minimizing false alarms and maintaining user trust. The respectable recall shows the model's ability to catch a substantial proportion of actual frauds, reducing the risk of missed threats. The combination of these metrics into a high F1 score underscores the model's balanced performance in both aspects.<br>
Moreover, the high ROC AUC score of 99.03% reflects the model's ability to distinguish between fraudulent and non-fraudulent messages across various threshold settings. This robustness is vital for adapting the model to different operational scenarios and risk tolerance levels in a real-world application.<br>
Limitations and Future Work<br>
While the results are promising, there are several areas for improvement and further investigation:<br>
Feature Enhancement: Our current model primarily relies on text features derived from TF-IDF and SMS length. Incorporating additional features, such as metadata (e.g., sender information, time of day), could improve detection accuracy.<br>
Advanced Text Representations: Exploring more sophisticated text representation techniques like word embeddings (Word2Vec, GloVe) or contextual embeddings (BERT) might capture deeper semantic nuances and improve model performance.<br>
Next Step<br>
Model Optimization: Experiment with imbalanced data handling techniques and further hyperparameter tuning.<br>
Deployment and Monitoring: Develop a pipeline for real-time deployment, including monitoring and continuous learning from user feedback.<br>
