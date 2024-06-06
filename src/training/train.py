import argparse
import pandas as pd
from pathlib import Path
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from azureml.core import Run

sys.path.append(str(Path(__file__).parents[1]))
from utils.data_utils import load_data

def train(data, output_path):
    X_train, X_test, y_train, y_test = load_data(data)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"Model accuracy: {accuracy}")
    
    with open(f"{output_path}/model.pkl", "wb") as f:
        import pickle
        pickle.dump(model, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the trained model")
    args = parser.parse_args()
    
    run = Run.get_context()
    if hasattr(run, "input_datasets"):
        # If running in Azure ML, get the dataset as a DataFrame
        dataset = run.input_datasets['training_data']
        data = dataset.to_pandas_dataframe()
    else:
        # If running locally, read the CSV file
        data = pd.read_csv(f"{args.data_path}/train.csv")

    train(data, args.output_path)
