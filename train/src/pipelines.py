import sys
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression

sys.path.append('././')
from transformers import ReplaceQM
from transformers import CleanData
from transformers import AddColumns

class TitanicClassifierPipeline():
    def __init__(self):
        self.columns_dict = {
            'num':[
                'pclass',
                'age',
                'sibsp',
                'parch',
                'fare'
                ],
            'categories':[
                'sex',
                'embarked',
                'title',
                'cabin_letter'
                ],
            'skip':[
                'cabin',
                'boat',
                'body',
                'home.dest',
                'ticket',
                'name'
                ]
            }

        self.num_pipeline = Pipeline([
            (
                'imputer',
                SimpleImputer(strategy='median')
            ),
            (
                'clean_data',
                CleanData()
            )])

        self.category_pipeline = Pipeline([
            (
                'imputer',
                SimpleImputer(strategy='most_frequent')
            ),
            (
                'encoder',
                OneHotEncoder(
                    sparse = False,
                    drop = 'first'
            ))
        ])

        self.titanic_pipeline = Pipeline([
            (
                'replace_question_marks',
                ReplaceQM()
            ),
            (
                'add_columns',
                AddColumns()
            ),
            (
                'tranform_data',
                ColumnTransformer([
                    (
                        'number_columns',
                        self.num_pipeline,
                        self.columns_dict['num']
                    ),
                    (
                        'category_columns',
                        self.category_pipeline,
                        self.columns_dict['categories']
                    )
                ])
                
            ),
            (
                'scaler',
                StandardScaler()
            ),
            (
                'estimator',
                LogisticRegression()
            )
        ])