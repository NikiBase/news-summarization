import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
import numpy as np
from pathlib import Path


def cleaning(df: pd.DataFrame):
    # \r and \n
    df["Content_Parsed_1"] = df["Content"].str.replace("\r", " ")
    df["Content_Parsed_1"] = df["Content_Parsed_1"].str.replace("\n", " ")
    df["Content_Parsed_1"] = df["Content_Parsed_1"].str.replace("    ", " ")
    # " when quoting text
    df["Content_Parsed_1"] = df["Content_Parsed_1"].str.replace('"', "")
    # Lowercasing the text
    df["Content_Parsed_2"] = df["Content_Parsed_1"].str.lower()
    punctuation_signs = list("?:!.,;")
    df["Content_Parsed_3"] = df["Content_Parsed_2"]

    for punct_sign in punctuation_signs:
        df["Content_Parsed_3"] = df["Content_Parsed_3"].str.replace(punct_sign, "")

    df["Content_Parsed_4"] = df["Content_Parsed_3"].str.replace("'s", "")
    # Saving the lemmatizer into an object
    wordnet_lemmatizer = WordNetLemmatizer()
    nrows = len(df)
    lemmatized_text_list = []

    for row in range(0, nrows):

        # Create an empty list containing lemmatized words
        lemmatized_list = []

        # Save the text and its words into an object
        text = df.loc[row]["Content_Parsed_4"]
        text_words = text.split(" ")

        # Iterate through every word to lemmatize
        for word in text_words:
            lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))

        # Join the list
        lemmatized_text = " ".join(lemmatized_list)

        # Append to the list containing the texts
        lemmatized_text_list.append(lemmatized_text)
    df["Content_Parsed_5"] = lemmatized_text_list
    stop_words = list(stopwords.words("english"))
    df["Content_Parsed_6"] = df["Content_Parsed_5"]

    for stop_word in stop_words:
        regex_stopword = r"\b" + stop_word + r"\b"
        df["Content_Parsed_6"] = df["Content_Parsed_6"].str.replace(regex_stopword, "")
    return df


def label_coding(df: pd.DataFrame):
    category_codes = {
        "business": 0,
        "entertainment": 1,
        "politics": 2,
        "sport": 3,
        "tech": 4,
    }
    # Category mapping
    df["Category_Code"] = df["Category"]
    df = df.replace({"Category_Code": category_codes})
    return df


def data_split(df: pd.DataFrame):
    return train_test_split(
        df["Content_Parsed"], df["Category_Code"], test_size=0.15, random_state=8
    )


def text_representation(X_train, X_test, y_train, y_test):
    # Parameter election
    ngram_range = (1, 2)
    min_df = 10
    max_df = 1.0
    max_features = 300
    tfidf = TfidfVectorizer(
        encoding="utf-8",
        ngram_range=ngram_range,
        stop_words=None,
        lowercase=False,
        max_df=max_df,
        min_df=min_df,
        max_features=max_features,
        norm="l2",
        sublinear_tf=True,
    )

    features_train = tfidf.fit_transform(X_train).toarray()
    labels_train = y_train

    features_test = tfidf.transform(X_test).toarray()
    labels_test = y_test

    return tfidf, features_train, labels_train, features_test, labels_test


if __name__ == "__main__":
    # Downloading punkt and wordnet from NLTK
    nltk.download("punkt")
    print("------------------------------------------------------------")
    nltk.download("wordnet")
    print("------------------------------------------------------------")
    # Downloading the stop words list
    nltk.download("stopwords")
    print("------------------------------------------------------------")
    nltk.download("omw-1.4")
    print("------------------------------------------------------------")

    datapath = Path("data/News_dataset.csv")
    df = pd.read_csv(datapath, sep=";")
    df_clean = cleaning(df)
    list_columns = [
        "File_Name",
        "Category",
        "Complete_Filename",
        "Content",
        "Content_Parsed_6",
    ]
    df_clean = df_clean[list_columns]

    df_clean = df_clean.rename(columns={"Content_Parsed_6": "Content_Parsed"})

    df_label_coding = label_coding(df_clean)

    X_train, X_test, y_train, y_test = data_split(df_label_coding)

    (
        tfidf,
        features_train,
        labels_train,
        features_test,
        labels_test,
    ) = text_representation(X_train, X_test, y_train, y_test)
    # X_train
    with open("Pickles/X_train.pickle", "wb") as output:
        pickle.dump(X_train, output)

    # X_test
    with open("Pickles/X_test.pickle", "wb") as output:
        pickle.dump(X_test, output)

    # y_train
    with open("Pickles/y_train.pickle", "wb") as output:
        pickle.dump(y_train, output)

    # y_test
    with open("Pickles/y_test.pickle", "wb") as output:
        pickle.dump(y_test, output)

    # df
    with open("Pickles/df_label_coding.pickle", "wb") as output:
        pickle.dump(df_label_coding, output)

    # features_train
    with open("Pickles/features_train.pickle", "wb") as output:
        pickle.dump(features_train, output)

    # labels_train
    with open("Pickles/labels_train.pickle", "wb") as output:
        pickle.dump(labels_train, output)

    # features_test
    with open("Pickles/features_test.pickle", "wb") as output:
        pickle.dump(features_test, output)

    # labels_test
    with open("Pickles/labels_test.pickle", "wb") as output:
        pickle.dump(labels_test, output)

    # TF-IDF object
    with open("Pickles/tfidf.pickle", "wb") as output:
        pickle.dump(tfidf, output)
