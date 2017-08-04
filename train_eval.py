from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
import pandas as pd
import numpy as np

# This trains and evaluates the Titanic model for the Kaggle DS Titanic 
# competition

def extract_clean_data(data_file_name):
    """ Extracts and does some preliminary feature reduction on the dataset
    data_file_name: string of the filename of the .csv file with our training 
    data
    returns: a cleaned up pandas dataset
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
    num_df_filled = imp.fit_transform(num_df)

    # Convert numpy array back into a Pandas dataframe type
    num_df_filled = pd.DataFrame(num_df_filled, columns = num_df.columns)

    # Join the imputed values with the non-numerical values
    final_df = pd.concat(
        [
            num_df_filled, 
            data_frame.drop("Age", axis = 1)
        ], 
        axis = 1, join = "inner"
    )

    return final_df


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


def eval_test_data(model, test_df):
    """ Evaluates the provided test data to rank the effectiveness of the 
    training function
    model: the trained model to evaluate
    test_df: the dataframe containing the test data
    """
    pass # TODO


def write_pred(pred_df, filename):
    """ Writes the predictions obtained by the estimator to a CSV file as 
    specified by the filename
    pred_df: a dataframe containing the predictions created by the estimator
    filename: the desired filename to write the predictions to disk 
    """
    pass # TODO


def main():
    """Wrapper for main function
    """
    # Constants
    y_label = "Survived"

    # Load and clean training data
    print("Loading, cleaning, slicing training data...")
    training_data = extract_clean_data("train.csv")
    print("Done")

    # Printing info about training data to user
    print("Feature training dataframe shape:")
    print(training_data.count())
    print(training_data)

    # train the model
    estimator = train_model_rf(training_data, y_label)


# Wrapper for main function
if __name__ == "__main__":
    main()


