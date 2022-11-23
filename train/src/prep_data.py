import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

import json
import requests

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression



DATASETS_DIR = 'datasets/'
URL = 'https://www.openml.org/data/get_csv/16826755/phpMYEkMl'
DROP_COLS = ['boat','body','home.dest','ticket','name']
RETRIEVED_DATA = 'raw-data.csv'


SEED_SPLIT = 404

TRAIN_DATA_FILE = f'{DATASETS_DIR}train.csv'
TEST_DATA_FILE = f'{DATASETS_DIR}test.csv'


TARGET = 'survived'
FEATURES = ['pclass','sex','age','sibsp','parch','fare','cabin','embarked','title']
NUMERICAL_VARS = ['pclass','age','sibsp','parch','fare']
CATEGORICAL_VARS = ['sex','cabin','embarked','title']

NUMERICAL_VARS_WITH_NA = ['age','fare']
CATEGORICAL_VARS_WITH_NA = ['cabin','embarked']
NUMERICAL_NA_NOT_ALLOWED = [var for var in NUMERICAL_VARS if var not in NUMERICAL_VARS_WITH_NA]
CATEGORICAL_NA_NOT_ALLOWED = [var for var in CATEGORICAL_VARS if var not in CATEGORICAL_VARS_WITH_NA]

SEED_MODEL = 404

def data_retrieval(url):
     
    # Loading data from specific url
    data = pd.read_csv(url)
    
    # Uncovering missing data
    data.replace('?', np.nan, inplace=True)
    data['age'] = data['age'].astype('float')
    data['fare'] = data['fare'].astype('float')
    
    # helper function 1
    def get_first_cabin(row):
        try:
            return row.split()[0]
        except:
            return np.nan
    
    # helper function 2
    def get_title(passenger):
        line = passenger
        if re.search('Mrs', line):
            return 'Mrs'
        elif re.search('Mr', line):
            return 'Mr'
        elif re.search('Miss', line):
            return 'Miss'
        elif re.search('Master', line):
            return 'Master'
        else:
            return 'Other'
    
    # Keep only one cabin | Extract the title from 'name'
    data['cabin'] = data['cabin'].apply(get_first_cabin)
    data['title'] = data['name'].apply(get_title)
    
    # Droping irrelevant columns
    data.drop(columns=DROP_COLS, 1, inplace=True)
    
    data.to_csv(DATASETS_DIR + RETRIEVED_DATA, index=False)
    
    return print('Data stored in {}'.format(DATASETS_DIR + RETRIEVED_DATA))

    class MissingIndicator(BaseEstimator, TransformerMixin):
    
        def __init__(self, variables=None):
            if not isinstance(variables, list):
                self.variables = [variables]
            else:
                self.variables = variables
    
        def fit(self, X, y=None):
            return self
        
        def transform(self, X):
            X = X.copy()
            for var in self.variables:
                X[var+'_nan'] = X[var].isnull().astype(int)
            
            return X
    

class ExtractLetters(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.variable = 'cabin'
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X[self.variable] = X[self.variable].apply(lambda x: ''.join(re.findall("[a-zA-Z]+", x)) if type(x)==str else x)
        return X
    

class CategoricalImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self, variables=None):
        self.variables = variables if isinstance(variables, list) else [variables]
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var] = X[var].fillna('Missing')
        return X

    
class NumericalImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self, variables=None):
        self.variables = variables if isinstance(variables, list) else [variables]
    
    def fit(self, X, y=None):
        self.median_dict_ = {}
        for var in self.variables:
            self.median_dict_[var] = X[var].median()
        return self
        

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var] = X[var].fillna(self.median_dict_[var])
        return X

    
class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, tol=0.05, variables=None):
        self.tol = tol
        self.variables = variables if isinstance(variables, list) else [variables]
                    
    def fit(self, X, y=None):
        self.rare_labels_dict = {}
        for var in self.variables:
            t = pd.Series(X[var].value_counts() / np.float(X.shape[0]))
            self.rare_labels_dict[var] = list(t[t<self.tol].index)
        return self
    
    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var] = np.where(X[var].isin(self.rare_labels_dict[var]), 'rare', X[var])
        return X


class OneHotEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, variables=None):
        self.variables = variables if isinstance(variables, list) else [variables]
       
    def fit(self, X, y=None):
        self.dummies = pd.get_dummies(X[self.variables], drop_first=True).columns
        return self
    
    def transform(self, X):
        X = X.copy()
        X = pd.concat([X, pd.get_dummies(X[self.variables], drop_first=True)], 1)
        X.drop(self.variables, 1, inplace=True)
        
        # Adding missing dummies, if any
        if missing_dummies := [var for var in self.dummies if var not in X.columns]:
             for col in missing_dummies:
                 X[col] = 0
        
        return X


class OrderingFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None
            
    def fit(self, X, y=None):
        self.ordered_features = X.columns
        return self
    
    def transform(self, X):
        return X[self.ordered_features]


# scaler = MinMaxScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test  = scaler.transform(X_test)

# model = LogisticRegression(C=0.0005, class_weight='balanced', random_state=SEED_MODEL)
# model.fit(X_train, y_train)

titanic_pipeline = Pipeline(
    [
    ('missing_indicator', MissingIndicator(variables=NUMERICAL_VARS)),
    ('cabin_only_letter', ExtractLetters()),
    ('categorical_imputer', CategoricalImputer(variables=CATEGORICAL_VARS_WITH_NA)),
    ('median_imputation', NumericalImputer(variables=NUMERICAL_VARS_WITH_NA)),
    ('rare_labels', RareLabelCategoricalEncoder(tol=0.05, variables=CATEGORICAL_VARS)),
    ('dummy_vars', OneHotEncoder(variables=CATEGORICAL_VARS)),
    ('aligning_feats', OrderingFeatures()),
    ('scaling', MinMaxScaler()),
    ('log_reg', LogisticRegression(C=0.0005, class_weight='balanced', random_state=SEED_MODEL))
    ])

df = pd.read_csv(DATASETS_DIR + RETRIEVED_DATA)

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(TARGET, axis=1),
    df[TARGET],
    test_size=0.2,
    random_state=404
)

titanic_pipeline.fit(X_train, y_train)

class_pred = titanic_pipeline.predict(X_test)
proba_pred = titanic_pipeline.predict_proba(X_test)[:,1]
print(f'test roc-auc : {roc_auc_score(y_test, proba_pred)}')
print(f'test accuracy: {accuracy_score(y_test, class_pred)}')
print()

TRAINED_MODEL_DIR = 'trained_models/'
PIPELINE_NAME = 'logistic_regression'
PIPELINE_SAVE_FILE = f'{PIPELINE_NAME}_output.pkl'

save_file_name = f'{PIPELINE_SAVE_FILE}'
save_path = TRAINED_MODEL_DIR + save_file_name

pipeline_to_persist = titanic_pipeline

# joblib.dump(pipeline_to_persist, save_path)

input_data = X_test.copy()

validated_data = input_data

if input_data[NUMERICAL_NA_NOT_ALLOWED].isnull().any().any():
    validated_data = validated_data.dropna(subset=NUMERICAL_NA_NOT_ALLOWED)
        
if input_data[CATEGORICAL_NA_NOT_ALLOWED].isnull().any().any():
    validated_data = validated_data.dropna(subset=CATEGORICAL_NA_NOT_ALLOWED)

file_path = TRAINED_MODEL_DIR + PIPELINE_SAVE_FILE
trained_model = joblib.load(filename=file_path)

preds = trained_model.predict(validated_data)
proba = trained_model.predict_proba(validated_data)

pd.concat([validated_data.reset_index(), pd.Series(preds, name='preds'), pd.Series(pd.DataFrame(proba)[1], name='probas')], 1).head()

#Predictions with the model served as REST API

url = 'http://127.0.0.1:5000/v1/predict/classification'
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

data1 = {"pclass":1,"sex":"male","age":58.0,"sibsp":0,"parch":2,"fare":113.275,"cabin":"D48","embarked":"C","title":"Mr"}
data2 = {"pclass":2,"sex":"male","age":31.0,"sibsp":1,"parch":1,"fare":26.25,"cabin":None,"embarked":"S","title":"Mr"}
data3 = {"pclass":3,"sex":"female","age":18.0,"sibsp":0,"parch":0,"fare":7.8792,"cabin":None,"embarked":"Q","title":"Miss"}
data4 = {"pclass":2,"sex":"male","age":34.0,"sibsp":1,"parch":0,"fare":21.0,"cabin":None,"embarked":"S","title":"Mr"}
data5 = {"pclass":2,"sex":"male","age":39.0,"sibsp":0,"parch":0,"fare":26.0,"cabin":None,"embarked":"S","title":"Mr"}

for d in [X_test[i:i+1].to_json(orient='records') for i in range(25)]:
    info = json.loads(d)[0]
    x = requests.post(url, data=json.dumps(info), headers=headers)
    print(x.json())

prueba = '{"pclass":1,"sex":"male","age":58.0,"sibsp":0,"parch":2,"fare":113.275,"cabin":"D48","embarked":"C","title":"Mr"}'
type(json.loads(prueba) )

tmp = pd.DataFrame(X_test, columns=list(sort_feats.ordered_features))
tmp['y_true'] = np.array(y_test)
tmp['y_pred'] = model.predict(X_test)
tmp['proba_pred'] = model.predict_proba(X_test)[:,1]

tmp.head(10)

