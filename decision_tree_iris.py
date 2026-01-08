# decision_tree_iris.py
# Decision Tree Classification - Iris Dataset

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

def main():
    # Load dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)

    # EDA singkat
    print("Contoh data:")
    print(X.head())
    print("\nDistribusi kelas:")
    print(y.value_counts())

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build model Decision Tree
    model = DecisionTreeClassifier(
        criterion='gini',
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)

    # Evaluasi
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Visualisasi Decision Tree
    plt.figure(figsize=(15, 8))
    plot_tree(
        model,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        filled=True
    )
    plt.title("Decision Tree - Iris Dataset")
    plt.show()

if __name__ == "__main__":
    main()
