from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

def train_model(X_train, y_train):
    """
    Trains a Random Forest Classifier on the given data.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model using accuracy and classification report.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return accuracy, report

# Optional: test the module
if __name__ == "__main__":
    from preprocessing import load_data, preprocess_data

    # Load and preprocess data
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    accuracy, report = evaluate_model(model, X_test, y_test)
    
    print("✅ Model training completed")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", report)
