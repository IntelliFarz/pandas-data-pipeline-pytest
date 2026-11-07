# Assignment 3: pytest test for clean()
# Name: Farzain Ahmed | Trainee ID: TRN-2026-001153
import pandas as pd

from pipeline import clean


def test_clean():
    messy = pd.DataFrame({
        "name": ["Ali", "Sara", "Omar", "Omar", None],
        "city": ["Karachi", " karachi", "KARACHI ", "KARACHI ", "Lahore"],
        "age": ["21", "22", "twenty", "twenty", "23"],
        "score": [80.0, 92.0, None, None, 66.0],
    })

    result = clean(messy)

    # the duplicate Omar row and the row with no name are removed
    assert result["name"].tolist() == ["Ali", "Sara", "Omar"]
    # spaces and capital letters in city are fixed
    assert result["city"].tolist() == ["Karachi", "Karachi", "Karachi"]
    # "twenty" becomes 20
    assert result["age"].tolist() == [21, 22, 20]
    # Omar's missing score is filled with the median of 80 and 92
    assert result["score"].tolist() == [80, 92, 86]
    # no missing values are left
    assert result.isna().sum().sum() == 0
