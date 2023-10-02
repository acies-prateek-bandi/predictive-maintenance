import logging
from typing import Dict, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder



def split_data_binary(data: pd.DataFrame, parameters: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    X = data[parameters["features"]]
    y = data["Target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    return X_train, X_test, y_train, y_test


def train_model_binary(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    binary_classifier = DecisionTreeClassifier(max_depth=10, min_samples_leaf=2, random_state=42)
    binary_classifier.fit(X_train, y_train)
    return binary_classifier

def split_data_multiclass(data: pd.DataFrame, parameters: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    data = data.drop(["Target",'Type'], axis=1, errors="ignore")

    X = data.drop("Failure Type", axis=1)
    y = data["Failure Type"]

    X_train_multi, X_test_multi, y_train, y_test = train_test_split(
    X, y, test_size=parameters["test_size"], random_state=parameters["random_state"])

    ohe = OneHotEncoder()
    y_train = ohe.fit_transform(y_train.values.reshape(-1, 1))
    y_test = ohe.transform(y_test.values.reshape(-1, 1))

    y_train_multi = pd.DataFrame(y_train.toarray(), columns=ohe.categories_)
    y_test_multi = pd.DataFrame(y_test.toarray(), columns=ohe.categories_)


    # logger = logging.getLogger(__name__)
    # logger.info("Multiclass Data")

    # logger.info("X_train shape: ", X_train_multi.shape)
    # logger.info("y_train shape: ", y_train_multi.shape)
    # logger.info("X_test shape: ", X_test_multi.shape)
    # logger.info("y_test shape: ", y_test_multi.shape)

    return X_train_multi, X_test_multi, y_train_multi, y_test_multi




def train_model_multiclass(X_train_multi: pd.DataFrame, y_train_multi: pd.Series) -> DecisionTreeClassifier:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    multiclass_classifier = RandomForestClassifier(max_depth=15, min_samples_leaf=2, random_state=42)
    multiclass_classifier.fit(X_train_multi, y_train_multi)
    return multiclass_classifier







def evaluate_model(
    model, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, float]:
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="micro")
    recall = recall_score(y_test, y_pred, average="micro")
    f1 = f1_score(y_test, y_pred, average="micro")

    logger = logging.getLogger(__name__)
    logger.info("Multi class classifier has an accuracy of %.3f on test data.", accuracy)
    logger.info("Multi class classifier has a precision of %.3f on test data.", precision)
    logger.info("Multi class classifier has a recall of %.3f on test data.", recall)
    logger.info("Multi class classifier has f1 score of %.3f on test data.", f1)

    return {"accuracy": accuracy, "precision": precision, "recall": recall,"f1":f1}
