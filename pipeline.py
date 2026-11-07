# ============================================================
# Assignment 3: Build your own pipeline (load -> clean -> add_features)
# Name       : Farzain Ahmed
# Trainee ID : TRN-2026-001153
# Course     : Artificial Intelligence
# Cohort     : PSEB-AI-KHI-01 (PSEB-AI-KHI-02)
# Trainer    : Lajvanti Ratnani
# ============================================================
import pandas as pd

# number words, so that an age written as "twenty" becomes 20
UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
         "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
         "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
         "seventeen": 17, "eighteen": 18, "nineteen": 19}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
        "seventy": 70, "eighty": 80, "ninety": 90}


def age_to_number(value):
    """Turn "21" or "twenty one" into a number; anything else becomes NaN."""
    if pd.isna(value):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    total = 0
    for word in str(value).lower().replace("-", " ").split():
        if word in UNITS:
            total += UNITS[word]
        elif word in TENS:
            total += TENS[word]
        else:
            return float("nan")
    return float(total)


def load(path="students_messy.csv"):
    """Step 1: load the messy data from a CSV file."""
    return pd.read_csv(path)


def clean(df):
    """Step 2: clean the data (returns a new DataFrame)."""
    df = df.copy()

    # 1. fix text columns: remove extra spaces and use Title Case
    df["name"] = df["name"].astype("string").str.strip().str.title()
    df["city"] = df["city"].astype("string").str.strip().str.title()

    # 2. drop rows with a missing or empty name
    df = df.dropna(subset=["name"])
    df = df[df["name"] != ""]

    # 3. make age and score numeric ("twenty" -> 20)
    df["age"] = df["age"].apply(age_to_number)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    # 4. remove duplicate rows
    df = df.drop_duplicates()

    # 5. fill missing age and score with the median value
    df["age"] = df["age"].fillna(df["age"].median())
    df["score"] = df["score"].fillna(df["score"].median())

    # 6. use whole numbers for age and score
    df["age"] = df["age"].astype(int)
    df["score"] = df["score"].astype(int)

    return df.reset_index(drop=True)


def add_features(df):
    """Step 3: add a new feature: passed (True when the score is 50 or more)."""
    df = df.copy()
    df["passed"] = df["score"] >= 50
    return df


def main():
    print("Name: Farzain Ahmed | Trainee ID: TRN-2026-001153")
    print("-" * 55)

    df = load("students_messy.csv")
    print(f"Loaded students_messy.csv: {df.shape[0]} rows, {df.shape[1]} columns")

    df = clean(df)
    print(f"After clean():             {df.shape[0]} rows, {df.shape[1]} columns")

    df = add_features(df)
    print(f"After add_features():      {df.shape[0]} rows, {df.shape[1]} columns")

    df.to_csv("students_clean.csv", index=False)
    print("Saved students_clean.csv\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
