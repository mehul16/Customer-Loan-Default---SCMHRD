
# import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix


loan = pd.read_csv("Train_Data.csv")
loan.set_index("ID", inplace=True)
loan.reset_index(drop=True, inplace=True)
loan.head()



loan.dropna(inplace=True)
loan.isnull().sum()





# # Changed
# # This is stratified sampling based on "Credit History".
# split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.3, random_state = 42)
# for train_index_s, test_index_s in split.split(loan, loan["Loan_Status"]):
#     strat_train_set = loan.iloc[train_index_s]
#     strat_test_set = loan.iloc[test_index_s]



# This is stratified sampling based on "Credit History".
split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_index_s, test_index_s in split.split(loan, loan["Credit_History"]):
    strat_train_set = loan.iloc[train_index_s]
    strat_test_set = loan.iloc[test_index_s]



strat_train_set.head()



# # seperate the numerical & categorical features
# train_num_col  = list(strat_train_set.select_dtypes(["int64","float"]).columns)
# train_cat_col = list(strat_train_set.select_dtypes("object").columns)



# print(train_num_col)
# print(train_cat_col)



# train_num = strat_train_set[train_num_col]
# train_cat = strat_train_set[train_cat_col]


# strat_train_set["Income_of_Applicant"] = pd.to_numeric(strat_train_set["Income_of_Applicant"], downcast = "float")


# train_num.head(10)


# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(strategy="median")



# imputer.fit(train_num)
# print(imputer.statistics_)



# train_num_imp = pd.DataFrame(imputer.transform(train_num), columns = train_num.columns, index = train_num.index)
# train_num_imp.head(5)





# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()


# scaler.fit(train_num_imp)
# train_num_tr = scaler.transform(train_num_imp)



# train_num_tr = pd.DataFrame(scaler.transform(train_num_imp), columns = train_num.columns, index = train_num.index)
# train_num_tr.head(5)



# import sys
# sys.exit()



def mm_scaler(dataset):
    mmscaler = MinMaxScaler()
    data_scaled = mmscaler.fit_transform(dataset)
    return data_scaled


# # This is simple random sampling.
# np.random.seed(42)
# shuffles = np.random.permutation(len(loan))

# test_ratio = 0.7
# train_set_size = int(len(loan) * test_ratio)

# train_index = shuffles[:train_set_size]
# test_index = shuffles[train_set_size:]

# strat_train_set = loan.iloc[train_index]
# strat_test_set = loan.iloc[test_index]

strat_train_set.head()


train_loan_x = strat_train_set.drop("Loan_Status", axis=1)
train_loan_y = strat_train_set["Loan_Status"]


test_loan_x = strat_test_set.drop("Loan_Status", axis=1)
test_loan_y = strat_test_set["Loan_Status"]




# Label encoding & 1-hot encoding
def binarize_1hot(dataset, to_binarize, to_one_hot):
    for i in to_binarize:
        dataset[i] = LabelBinarizer().fit_transform(dataset[i])
    dataset = pd.get_dummies(data=dataset, columns=to_one_hot)
    return dataset



# Min-Max Scaler function
def mm_scaler(dataset):
    mmscaler = MinMaxScaler()
    data_scaled = mmscaler.fit_transform(dataset)
    return data_scaled


# label encoding
def label_encode(dataset):
    label_enc = LabelBinarizer()
    dataset_encoded = label_enc.fit_transform(dataset)
    return dataset_encoded



# label encoding & one-hot encoding train_X data
to_binarize = ["Gender", "Is_Married",
               "Level_of_Education", "IS_Self_Employed"]
to_one_hot = ["No_of_Dependents", "Area_of_Property"]

train_x_encoded = binarize_1hot(train_loan_x, to_binarize, to_one_hot)
test_x_encoded = binarize_1hot(test_loan_x, to_binarize, to_one_hot)




train_x_encoded_scaled = mm_scaler(train_x_encoded)
test_x_encoded_scaled = mm_scaler(test_x_encoded)




train_y_encoded_scaled = label_encode(train_loan_y)
test_y_encoded_scaled = label_encode(test_loan_y)



# Training a Logistic Regression model
LR_model = LogisticRegression(C=0.5).fit(
    train_x_encoded_scaled, train_y_encoded_scaled)

# Testing on test set
yhat = LR_model.predict(test_x_encoded_scaled)
print("LR Jaccard index: %f" % jaccard_score(
    test_y_encoded_scaled, yhat, pos_label=1))
print("LR F1-score: %f" %
      f1_score(test_y_encoded_scaled, yhat, average='weighted', pos_label=1))
print(confusion_matrix(test_y_encoded_scaled, yhat))



# Random sampling at 70:30 split.

# LR Jaccard index: 0.784000
# LR F1-score: 0.786952
# array([[19, 26],
#        [ 1, 98]], dtype=int64)


# Stratified sampling using "Credit_History".

# LR Jaccard index: 0.801587
# LR F1-score: 0.806617
# array([[ 18,  22],
#        [  3, 101]], dtype=int64)
