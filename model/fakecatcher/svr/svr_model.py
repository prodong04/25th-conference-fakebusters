import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

class SVRModel:
    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1):
        """
        Initialize the SVR model with the specified parameters.

        Parameters:
        - kernel: Kernel type (e.g., 'linear', 'poly', 'rbf'). Default is 'rbf'.
        - C: Regularization parameter. Default is 1.0.
        - epsilon: Epsilon in the epsilon-SVR model. Default is 0.1.
        """
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.model = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon)

    def train(self, X_train, y_train):
        """
        Train the SVR model on the training data.

        Parameters:
        - X_train: Training features.
        - y_train: Training target values.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """
        Predict target values using the trained SVR model.

        Parameters:
        - X: Features for prediction.

        Returns:
        - Predicted target values.
        """
        return self.model.predict(X)

    def evaluate(self, y_true, y_pred):
        """
        Evaluate the model's performance using Mean Squared Error (MSE).

        Parameters:
        - y_true: True target values.
        - y_pred: Predicted target values.

        Returns:
        - mse: Mean Squared Error of the predictions.
        """
        mse = mean_squared_error(y_true, y_pred)
        return mse

    def save_model(self, model_path):
        """모델과 스케일러를 파일로 저장합니다."""
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """저장된 모델과 스케일러를 로드합니다."""
        self.model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")

if __name__ == "__main__":
    # Example usage
    # Generate synthetic data
    X = np.random.rand(100, 1) * 10
    y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

    # Initialize the SVR model
    svr_model = SVRModel(kernel='rbf', C=1.0, epsilon=0.1)

    # Preprocess data
    X_train, X_test, y_train, y_test = svr_model.preprocess_data(X, y)

    # Train the model
    svr_model.train(X_train, y_train)

    # Predict on test data
    y_pred = svr_model.predict(X_test)

    # Evaluate the model
    mse = svr_model.evaluate(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")