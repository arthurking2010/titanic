import sys

import pytest

sys.path.append('./') 
from serve.src.functions import get_title

def get_title_test():
    return [
            ('Master Juan', 'Master'),
            ('Mr Juan', 'Mr'),
            ('Miss Juana', 'Miss'),
            ('Mrs Juana', 'Mrs'),
            ('Sor Juana', 'Other')
            ]

@pytest.mark.parametrize('passanger, title', get_title_test())
def test_get_title(passanger, title):
    assert get_title(passanger) == title

@pytest.mark.xfail
def test_get_title_fail():
    assert get_title('mr Juan') == 'Mr'