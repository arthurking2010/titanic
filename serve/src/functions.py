import re

def get_title(passenger:str) -> str:
    """ Return the title of a passenger name.
    It searches for the title of the name of a person:
    [Mrs, Mr, Miss, Master, Other] and returns the title.

    Parameters
    ----------
    passenger : string
        The name of the passenger including the title of the person.

    Returns
    -------
    string
        The title of the passenger.
    """ 
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