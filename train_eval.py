from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Imputer
import pandas as pd
import numpy as np

# This trains and evaluates the Titanic model for the Kaggle DS Titanic 
# competition

def extract_clean_data(data_file_name):
    """ Extracts and does some preliminary feature reduction on the dataset
    data_file_name: string of the filename of the .csv file with our training 
    data
    returns: a cleaned up pandas dataframe
    """
    data_frame =  pd.read_csv(data_file_name) 

    # Cleaning up data frame, dropping unecessary columns
    data_frame = data_frame.drop(
        [
            "PassengerId", 
            "Name",
            "Ticket",
            "Cabin", # TODO strip and factorize
        ], 
        axis = 1
    )

    factor_cols = ["Embarked", "Sex"]
    # Convert categorical non-numerical values into enumerated integers
    data_frame[factor_cols] = factor_column(data_frame, factor_cols)

    # Some age values are missing, replace them with median age of dataset
    # the imputer requires a data frame with only numerical values, the 
    # new dataframe drops all of the non-numerical values
    num_df = data_frame.drop(
        [
            "Embarked",
            "Sex",
        ],
        axis = 1
    )

    imp = Imputer(strategy = "median")
    impute_cols = [
        "Age",
        "Fare",
    ]
    data_frame[impute_cols] = imp.fit_transform(data_frame[impute_cols])
    return data_frame


def split_y_label(df, y_label):
    """ Splits a data frame into a set of features and a y label axis 
    df: the dataframe to split
    y_label: the name of the column to split off as the y
    returns (X, y) where X is the data frame with features and y is the 
    data frame to train on 
    """
    features = df.drop(y_label, axis = 1).columns
    X = df[features]
    y = df[y_label]
    return (X, y)

def train_model_rf(train_data, tr_col_name):
    """Trains a random forest estimator on a set of test data 
    test_data: a pandas dataframe with the data we want to use for training
    tr_col_name: the name of the column in the training dataframe that will 
    be used to fit the random forest classifier
    """
    # Create a random forest classifier
    # -1 sets num jobs to num cores
    rf_classifier = RandomForestClassifier(n_jobs = -1) 

    # Extract the features to be used for training
    features = train_data.drop(tr_col_name, axis = 1).columns
    rf_classifier.fit(train_data[features], train_data[tr_col_name])
    return rf_classifier


def factor_column(df, col_names):
    """helper function that factorizes the column(s) specified for a data 
    frame. Returns the dataframe with the columns factorized
    """
    return df[col_names].apply(lambda x: pd.factorize(x)[0])


def train_model_mlp(train_data, tr_col_name):
    """Trains a multi layer perceptron/neural net on a set of test data
    train_data: a pandas dataframe with the data to be used for training
    tr_col_name: the target classification to be used for training
    returns: a trained model
    """
    # Split off the labels for the training data 
    (X, y) = split_y_label(train_data, tr_col_name)
    clf = MLPClassifier()
    clf.fit(X, y)
    return clf


def predict_model(model, test_df, features):
    """ Evaluates the provided test data to rank the effectiveness of the 
    training function
    model: the trained model to evaluate
    test_df: the dataframe containing the test data
    """
    return model.predict(test_df[features])


def load_test_data(filename, factor_col_names):
    """ Loads, factorizes, cleans test data frame
    filename: the file name of the test data set to load
    factor_col_names: the name of the columns to factorie
    """
    # Loading pandas dataframe
    imp = Imputer(strategy = "median")
    test = pd.read_csv(filename)
    age_col = test["Age"]
    test[["Age", "Fare"]] = imp.fit_transform(test[["Age", "Fare"]])
    # Factoring non-numerical columns
    test[factor_col_names] = factor_column(test, factor_col_names)
    # Cabin has NaN values, which we can't have in a random forest
    test = test.drop("Cabin", axis = 1) 
    return test


def write_pred(pred_df, test_df, filename, id_col, data_col):
    """ Writes the predictions obtained by the estimator to a CSV file as 
    specified by the filename
    pred_df: a dataframe containing the predictions created by the estimator
    test_df: the testing dataframe
    filename: the desired filename to write the predictions to disk 
    """
    # Append id column from test dataset to predictions for submission
    pred_df[id_col] = test_df[id_col]

    # Specify order of columns/format
    csv = pred_df.to_csv(
        path_or_buf = filename, 
        index = False,
        columns = [id_col, data_col],
    )


def main():
    """Wrapper for main function
    """
    # Constants
    y_label = "Survived"
    factor_cols = ["Embarked", "Sex"]

    # Load and clean training data
    print("Loading, cleaning training data...") 
    training_data = extract_clean_data("train.csv")
    print("Done")
    print()

    # These are the features we will feed back into the estimator to 
    # yield predictions
    features = training_data.drop(y_label, axis = 1).columns
    # train the rf model
    print("Training random forest classifier with test data...")
    estimator = train_model_rf(training_data, y_label)
    print("Done")
    print()
    
    # test the irf model on the testing data
    print("Loading test data...")
    test_df = load_test_data("test.csv", factor_cols)
    print("Done")
    print()

    ####### RANDOM FOREST ######
    print("--Random forest classifier--")
    print()

    # Create predictions from the rf model using the testing data
    print("Making predictions with RF model...")
    predictions = predict_model(estimator, test_df, features) 

    # Convert prediction nparray to pandas dataframe
    predict_df = pd.DataFrame(
        data = predictions,
        columns = ["Survived"],
    )
    print("Done")
    print()

    # Write predictions to a CSV file based on the ID
    print("Writing random forest predictions to csv...")
    write_pred(predict_df, test_df, "model_output_rf.csv", 
               "PassengerId", "Survived")
    print("Done")
    print()

    ###### Multilayer perceptron algorithm with backpropagation ######
    print("--MLP backprop--")
    print()

    # train the MLP net
    print("Training MLP...")
    mlp_model = train_model_mlp(training_data, y_label)
    print("Done")
    print()

    # Create predictions with the MLP net
    print("Making predictions with the MLP...")
    mlp_predictions = predict_model(mlp_model, test_df, features)
    print("Done")
    print()

    # Convert predictions from multilayer perceptron to a pandas array
    mlp_predictions = pd.DataFrame(
        data = mlp_predictions,
        columns = ["Survived"],
    )

    # Write MLP predictions to csv file
    print("Writing MLP predictions to csv...")
    write_pred(mlp_predictions, test_df, "model_output_mlp.csv", 
               "PassengerId", "Survived")
    print("Done")
    print()
    

# Wrapper for main function
if __name__ == "__main__":
    main()

