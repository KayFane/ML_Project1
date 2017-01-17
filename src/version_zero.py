import pandas as pd
import numpy as np

# Load data
# Id	Name	Age Category	Sex	Rank	Time	Pace	Year
df_raw = pd.read_csv('../data/Project1_data.csv')

print(df_raw.info())

print(df_raw.describe())


def spread_table(df_org, col):
    if col == '':
        df_tmp = df_org[['Id', 'Year']].copy()
    else:
        df_tmp = df_org[['Id', 'Year', col]].copy()

    years = df_tmp['Year'].unique()

    # Add columns 2003, ..., 2016, indicating if the runner's record at each year
    for year in years:
        if col == '':
            df_tmp[str(year)] = (df_tmp['Year'] == year).astype(int)
        else:
            df_tmp[str(year)] = (df_tmp['Year'] == year).astype(int) * df_tmp[col]
    # Reduce duplicated rows
    df_rsl = df_tmp.drop('Year', 1)
    df_rsl = pd.pivot_table(df_rsl, index=['Id'], aggfunc=np.max)

    print(df_rsl)
    print(df_rsl.describe())

    return df_rsl


df_years = spread_table(df_raw, '')
df_years.to_csv("Id_Years.csv")

df_ranks = spread_table(df_raw, 'Rank')
df_ranks.to_csv("Id_Ranks.csv")


# Prepare Training Data


def extract_training_set(df_org):
    """
    Delete the rows with no running history before 2016;
    The training set is designed to predict 2016;
    No pre-running means not in the data set at the time of year 2016;
    """
    cols_of_interest = list(np.arange(2003, 2015).astype(str))
    df_rsl = df_org[(df_org[cols_of_interest] != 0).any(axis=1)]
    return df_rsl


df_training = extract_training_set(df_years)

print(df_training)

# fit a logistic regression model and store the class predictions
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics

model_logreg = LogisticRegression(C=1e9)

feature_cols = list(np.arange(2003, 2015).astype(str))

X = df_training[feature_cols]
y = df_training['2016']
model_logreg.fit(X, y)

print(model_logreg)
# make predictions
expected = y
predicted = model_logreg.predict(X)
# summarize the fit of the model

print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model2 = LogisticRegression()
model2.fit(X_train, y_train)
predicted = model2.predict(X_test)
print(metrics.classification_report(y_test, predicted))
print(metrics.confusion_matrix(y_test, predicted))



from sklearn.naive_bayes import GaussianNB
# fit a Naive Bayes model to the data
model_naive_bayes = GaussianNB()
model_naive_bayes.fit(X, y)
print(model_naive_bayes)
# make predictions
expected = y
predicted = model_naive_bayes.predict(X)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))