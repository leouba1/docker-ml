from sklearn.model_selection import train_test_split

def load_data(data):
    # Assuming 'data' is a DataFrame
    df = data
    df = df.drop(['PassengerId'], axis=1)  # Adjust based on your dataset columns
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    return train_test_split(X, y, test_size=0.2, random_state=42)