# By: Prakhar Prasun

from google.colab import drive
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class HousingAnalysis:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.df_encoded = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.svm_model = SVC(kernel='linear')

    def load_data(self):
        # Mounting Google Drive and loading data
        drive.mount('/content/drive')
        self.data = pd.read_csv(self.data_path)
        print(self.data.head(10))
        print(self.data.info())

    def preprocess_data(self):
        # One-hot encoding for non-numeric columns and scaling numeric columns
        df = pd.DataFrame(self.data)
        non_numeric_cols = df.select_dtypes(exclude=['number']).columns

        df_encoded = pd.get_dummies(df, columns=non_numeric_cols, drop_first=True)
        bool_cols = df_encoded.select_dtypes(include='bool').columns
        df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

        # Creating new features
        df_encoded['PriceperArea'] = df_encoded['price'] / df_encoded['area']
        df_encoded['TotalRooms'] = (
            df_encoded['bedrooms'] + df_encoded['bathrooms'] +
            df_encoded['parking'] + df_encoded['guestroom_yes'] + df_encoded['basement_yes']
        )

        # Scaling numeric columns
        numeric_cols = df_encoded.select_dtypes(include=['number']).columns
        scaler = StandardScaler()
        df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])
        
        self.df_encoded = df_encoded

    def prepare_data(self):
        # Preparing training and test sets
        X = self.df_encoded.drop('prefarea_yes', axis=1)
        y = pd.DataFrame(self.data['prefarea'].replace({'no': 0, 'yes': 1}))

        split_1 = int(0.6 * len(X))
        split_2 = int(0.8 * len(X))

        self.X_train, self.X_val, self.X_test = X[:split_1], X[split_1:split_2], X[split_2:]
        self.y_train, self.y_val, self.y_test = y[:split_1], y[split_1:split_2], y[split_2:]

    def train_svm(self):
        # Training the SVM model
        self.svm_model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        # Predicting and evaluating the model on training, validation, and test sets
        y_train_pred = self.svm_model.predict(self.X_train)
        y_val_pred = self.svm_model.predict(self.X_val)
        y_test_pred = self.svm_model.predict(self.X_test)

        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        val_accuracy = accuracy_score(self.y_val, y_val_pred)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)

        print(f'Train Accuracy: {train_accuracy * 100:.2f}%')
        print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')
        print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

    def plot_features_vs_price(self):
        # Plotting features against price
        for feature in self.df_encoded.columns:
            plt.figure(figsize=(16, 9))
            plt.scatter(self.df_encoded[feature], self.df_encoded['price'])
            plt.title(f'{feature} vs Price')
            plt.show()

    def plot_svm_decision_boundary(self):
        # Plotting SVM decision boundary for each feature
        w = self.svm_model.coef_[0]
        b = self.svm_model.intercept_[0]

        for col in self.X_train.columns:
            plt.figure(figsize=(16, 9))
            sns.scatterplot(self.X_train[col])

            # Creating decision boundary
            x_points = self.X_train['area']
            y_points = -(w[0] / w[1]) * x_points - b / w[1]
            plt.plot(x_points, y_points, color='red')
            plt.title(f'SVM Decision Boundary for {col}')
            plt.show()

# Usage
data_path = '/content/drive/MyDrive/ML Datasets for induction tasks/Housing.csv'
analysis = HousingAnalysis(data_path)

# Load and preprocess data
analysis.load_data()
analysis.preprocess_data()

# Prepare data for training and evaluation
analysis.prepare_data()

# Train SVM and evaluate
analysis.train_svm()
analysis.evaluate_model()

# Plotting
analysis.plot_features_vs_price()
analysis.plot_svm_decision_boundary()
