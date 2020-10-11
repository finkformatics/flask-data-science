import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List


class TextProcessing:
    def initialize_libraries(self) -> None:
        nltk.download('wordnet')
        nltk.download('stopwords')

    def _remove_special_chars(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Text_Processed'] = df['Text'].str.replace("\r", " ")
        df['Text_Processed'] = df['Text_Processed'].str.replace("\n", " ")
        df['Text_Processed'] = df['Text_Processed'].str.replace('"', '')

        return df

    def _to_lower_case(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Text_Processed'] = df['Text_Processed'].str.lower()

        return df

    def _remove_punctuation(self, df: pd.DataFrame) -> pd.DataFrame:
        punctuation_signs = list("?:!.,;-")

        for punct_sign in punctuation_signs:
            df['Text_Processed'] = df['Text_Processed'].str.replace(punct_sign, '')

        return df

    def _process_possessive_pronouns(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Text_Processed'] = df['Text_Processed'].str.replace("'s", "")

        return df

    def _lemmatize_words(self, df: pd.DataFrame) -> pd.DataFrame:
        wordnet_lemmatizer = WordNetLemmatizer()

        nrows = len(df)
        lemmatized_text_list = []
        for row in range(nrows):
            # Create an empty list containing lemmatized words
            lemmatized_list = []

            # Save the text and its words into an object
            text = df.loc[row]['Text_Processed']
            text_words = text.split()

            # Iterate through every word to lemmatize
            for word in text_words:
                lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))

            # Join the list
            lemmatized_text = " ".join(lemmatized_list)

            # Append to the list containing the texts
            lemmatized_text_list.append(lemmatized_text)

        df['Text_Processed'] = lemmatized_text_list

        return df

    def _remove_stop_words(self, df: pd.DataFrame) -> pd.DataFrame:
        stop_words = list(stopwords.words('english'))

        for stop_word in stop_words:
            regex_stopword = r"\b" + stop_word + r"\b"
            df['Text_Processed'] = df['Text_Processed'].str.replace(regex_stopword, '')

        return df

    def pre_process_text_list(self, texts: List[str]) -> pd.DataFrame:
        df = pd.DataFrame({'Text': texts})

        return self.pre_process_text_df(df)

    def pre_process_text_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._remove_stop_words(
            self._lemmatize_words(
                self._process_possessive_pronouns(
                    self._remove_punctuation(
                        self._to_lower_case(
                            self._remove_special_chars(df)
                        )
                    )
                )
            )
        )

        df['Text_Processed'] = df['Text_Processed'].str.replace("      ", " ")
        df['Text_Processed'] = df['Text_Processed'].str.replace("     ", " ")
        df['Text_Processed'] = df['Text_Processed'].str.replace("    ", " ")
        df['Text_Processed'] = df['Text_Processed'].str.replace("   ", " ")
        df['Text_Processed'] = df['Text_Processed'].str.replace("  ", " ")

        return df
