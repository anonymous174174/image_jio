import joblib

class ModelPredictor:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)

    def predict(self, features: list):
        prediction = self.model.predict([features])[0]
        probabilities = self.model.predict_proba([features])[0]
        return {
            "prediction": int(prediction),
            "probabilities": probabilities.tolist()
        }

if __name__ == "__main__":
    predictor = ModelPredictor("model/svm_model.pkl")
    sample_features = [0.0] * 64  # Example features for testing
    result = predictor.predict(sample_features)
    print(result)
