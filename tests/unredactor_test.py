import unredactor
import pandas as pd
import pytest

data = pd.read_table("unredactor.tsv",on_bad_lines='skip', sep='\t', names = ['GitID','type','labels','redacted_data'])

def test_check_length():
    if unredactor.check_length('█') == 1:
        assert True

def test_split_data():
    
    train_split, validation_split, test_split = unredactor.split_data(data)
    assert train_split.columns.tolist() == validation_split.columns.tolist() == test_split.columns.tolist() == data.columns.tolist()


if __name__ == "__main__":
    test_check_length()
    test_split_data()