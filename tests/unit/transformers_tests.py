import sys
import numpy as np

import pandas as pd

sys.path.append('./') 
from train.src.transformers import ReplaceQM
from train.src.transformers import CleanData
from train.src.transformers import AddColumns

df_to_test = pd.DataFrame(
    [
        ['Mr Juan', 'C1', '?'],
        ['Master Jose', 'B2', 25],
        ['Olivia', 'A3', 15]
    ],
    columns =
    [
        'name',
        'cabin',
        'age'
    ]
)

anwser_df_1 = pd.DataFrame(
    [
        ['Mr Juan', 'C1', np.nan],
        ['Master Jose', 'B2', 25],
        ['Olivia', 'A3', 15]
    ],
    columns =
    [
        'name',
        'cabin',
        'age'
    ]
)

anwser_df_3 = pd.DataFrame(
    [
        ['Mr Juan', 'C1', '?', 'Mr', 'C'],
        ['Master Jose', 'B2', 25, 'Master', 'B'],
        ['Olivia', 'A3', 15, 'Other', 'A']
    ],
    columns =
    [
        'name',
        'cabin',
        'age',
        'title',
        'cabin_letter'
    ]
)

def test_ReplaceQM():
    transformed_df = ReplaceQM().fit_transform(df_to_test)
    assert transformed_df.equals(anwser_df_1)

array_to_test = np.array(
    [
        [1.1, 1.2, 1.3, 1.4, 1.6],
        [2.1, 2.2, 2.3, 2.4, 2.6]
    ]
)

answer_array_2 = np.array(
    [
        [1.1, 1, 1.3, 1.4, 2.0],
        [2.1, 2, 2.3, 2.4, 3.0]
    ]
)

def test_AddColumns():
    transformed_df = AddColumns().fit_transform(df_to_test)
    assert transformed_df.equals(anwser_df_3)

def test_CleanData():
    transformed_array = CleanData().fit_transform(array_to_test)
    assert np.array_equal(transformed_array, answer_array_2)