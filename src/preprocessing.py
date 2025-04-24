import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(path="data/iris.csv"):
    """
    Loads the Iris dataset from a CSV file.
    """
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    """
    Preprocesses the Iris dataset:
    - Encodes the species labels
    - Separates features and target
    - Splits into train and test sets
    """
    # Encode target species
    le = LabelEncoder()
    df['species_encoded'] = le.fit_transform(df['species'])

    # Features and labels
    X = df.drop(['target', 'species', 'species_encoded'], axis=1)
    y = df['species_encoded']
