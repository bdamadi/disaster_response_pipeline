import sys
import pandas as pd
from sqlalchemy.engine import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load the messages dataset and the categories dataset from the
    given files and merge them to form the output dataset of the
    following columns:
        * original data from messages dataset
        * the corresponding categories (each per column) from
        the categories dataset.

    Parameters:
        messages_filepath (string) path to the messages.csv file
        categories_filepath (string) path to the categories.csv file
    Returns:
        a Pandas data frame of the messages data merged with 
        corresponding categories
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, on='id')

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    return pd.concat([df, categories], axis=1)

def clean_data(df): 
    """
    Clean up the given data frame to:
        Remove data rows with null message field
        Remove all duplicated rows

    Parameters:
        df (pd.DataFrame) input data frame
    Returns:
        The cleaned up data frame
    """   
    # Drop NA messages
    num_rows = df.shape[0]
    df = df.dropna(subset=['message'])
    print('    Dropped missing data : {}'.format(num_rows - df.shape[0]))

    # drop duplicates
    num_rows = df.shape[0]
    df = df.drop_duplicates()
    print('    Removed duplicated data : {}'.format(num_rows - df.shape[0]))

    return df

def save_data(df, database_filename):
    """
    Save the data frame into an SQLite database under 
    'DisasterResponse' table/
    
    Parameters:
        df (pd.DataFrame) input data frame
        database_filename (string) path to save the output SQLite database file
    Returns:
        None
    """
    # Save the clean dataset into an sqlite database
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


def main():
    """
    The main function to run this script to process input data files:
        messages data in CSV file
        categories data in CSV file
    Merge messages data and their corresponding categories, clean up,
    and then save to an SQLite database under DisasterResponse table.
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()