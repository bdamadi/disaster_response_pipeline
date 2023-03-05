import sys
import pandas as pd
import numpy as np
import pickle

from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

# Require to download the NLTK's packages.
# Un-comment these code if there are not downloaded yet.
# import nltk
# nltk.download('punkt')
# nltk.download('wordnet')

def load_data(database_filepath):
    """
    Load the DisasterResponse data from the given SQLite database file.
    Assume that the data are stored in 'DisasterResponse' table.

    Parameters:
        database_filepath (string) path to the SQLite database file
    Returns:
        A tuple of 3 values:
            A pandas.Series which contains the messages to be categorized
            A pandas.Dataframe of all categories of the corresponding message
            A list of category names
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)

    # Extract the message column
    X = df["message"]
    # Keep only category columns
    Y = df.drop(["id", "message", "original", "genre"], axis=1)
    return X, Y, Y.columns.values

def tokenize(text):
    """
    Tokenize the input text by the following steps:
        Split the text into words (using NLTK's word_tokenize)
        Lemmatize each word (using NLTK's WordNetLemmatizer)
        Normalize into lower case

    Parameters:
        text (string) input text to be tokenized
    Returns:
        List of tokenized words
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Build a pipeline to train a classifier to classify the disaster
    response messages into multi-categories.
    The pipeline contains the following steps:
        Count tokens from each input message
        Perform TF-IDF transform to the input text
        Build a MultiOutputClassifier over RandomForestClassifier
    """
    pipeline = Pipeline([
        ('count', CountVectorizer(tokenizer=tokenize)),
        ('tf-idf', TfidfTransformer()),
        ('classifier', MultiOutputClassifier(RandomForestClassifier()))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the classifier model using the input test dataset.
    Print out the classification report.

    Parameters:
        model: A multi-output classifier
        X_test: input test messages
        Y_test: input test categories
        category_names: list of category names (unused)
    Returns:
        None
    """
    Y_pred = model.predict(X_test)
    # Use np.hstack
    # https://stackoverflow.com/questions/56826216/valueerror-unknown-label-type-for-classification-report
    print(classification_report(np.hstack(Y_pred), np.hstack(Y_test.values)))


def save_model(model, model_filepath):
    """
    Save the classification model into a Pickle file.
    Parameters:
        model: the model to be saved
        model_filepath: path to save the model as an .pkl file
    Returns:
        None
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """
    The main function to run this script to train a multioutput
    classifier to classify disaster response message into the
    predefined categories.

    The input dataset is given as an SQLite database which is
    produced by the `process_data.py` script.

    Output is the classifier saved as a Pickle file.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()