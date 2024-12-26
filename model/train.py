from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

def train_model(model_path: str):
    print("Starting model training...")

    # Load digits dataset
    data = load_digits()
    X, y = data.data, data.target

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="linear", probability=True))
    ])

    # Train model
    pipeline.fit(X_train, y_train)
    print("Model trained successfully")

    # Evaluate model
    y_pred = pipeline.predict(X_test)
    print(f"Model accuracy: {accuracy_score(y_test, y_pred):.2f}")

    # Save model
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model("model/svm_model.pkl")
