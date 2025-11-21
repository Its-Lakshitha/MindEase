# src/preprocessing_pipeline.py
import os
import re
import pickle
from typing import List, Tuple, Any, Dict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split


# ------------------------------
# Helper: safe read CSVs concat
# ------------------------------
def _read_and_concat(raw_dir: str, prefix: str, file_range: List[int]) -> pd.DataFrame:
    dfs = []
    for i in file_range:
        path = os.path.join(raw_dir, f"{prefix}{i}.csv")
        dfs.append(pd.read_csv(path))
    df = pd.concat(dfs, ignore_index=True)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    return df


# ------------------------------
# Transformer: DatasetLoader
# ------------------------------
class DatasetLoader(BaseEstimator, TransformerMixin):
    def __init__(self, raw_dir: str, prefix: str = "emotion-recognition-", file_range: List[int] = None):
        self.raw_dir = raw_dir
        self.prefix = prefix
        self.file_range = list(file_range) if file_range is not None else list(range(1, 7))

    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None) -> pd.DataFrame:
        df = _read_and_concat(self.raw_dir, self.prefix, self.file_range)
        return df


# ------------------------------
# Transformer: TextCleaner
# ------------------------------
class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, text_column: str = "Text", out_column: str = "cleanedText"):
        self.text_column = text_column
        self.out_column = out_column

    @staticmethod
    def _clean(text: str) -> str:
        if pd.isna(text):
            return ""
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"@\S+", "", text)
        text = re.sub(r"#\S+", "", text)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.lower().strip()

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df[self.out_column] = df[self.text_column].astype(str).apply(self._clean)
        return df


# ------------------------------
# Transformer: EmotionSplitter
# ------------------------------
class EmotionSplitter(BaseEstimator, TransformerMixin):
    def __init__(self, emotion_column: str = "Emotion", out_column: str = "emotion_list", separator: str = None):
        """
        If separator is None -> split on whitespace after cleaning.
        If the emotion field uses commas, set separator=",".
        """
        self.emotion_column = emotion_column
        self.out_column = out_column
        self.separator = separator

    @staticmethod
    def _clean_emotion(e: str) -> str:
        if pd.isna(e):
            return ""
        e = re.sub(r"<.*?>", "", str(e))
        e = re.sub(r"\s+", " ", e)
        return e.strip().lower()

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if self.separator:
            df[self.out_column] = df[self.emotion_column].astype(str).apply(
                lambda s: [t.strip().lower() for t in self._clean_emotion(s).split(self.separator) if t.strip()]
            )
        else:
            df[self.out_column] = df[self.emotion_column].astype(str).apply(
                lambda s: [t.strip().lower() for t in self._clean_emotion(s).split() if t.strip()]
            )
        return df


# ------------------------------
# Transformer: MultiLabelEncoderTransformer
# ------------------------------
class MultiLabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, input_column: str = "emotion_list"):
        self.input_column = input_column
        self.mlb = MultiLabelBinarizer()

    def fit(self, X: pd.DataFrame, y=None):
        lists = X[self.input_column].tolist()
        self.mlb.fit(lists)
        return self

    def transform(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        df = X.copy()
        label_matrix = self.mlb.transform(df[self.input_column].tolist())
        # attach columns to df for convenience
        cols = list(self.mlb.classes_)
        label_df = pd.DataFrame(label_matrix, columns=cols, index=df.index)
        df = pd.concat([df, label_df], axis=1)
        return df, label_matrix

    def fit_transform(self, X: pd.DataFrame, y=None) -> Tuple[pd.DataFrame, np.ndarray]:
        self.fit(X)
        return self.transform(X)


# ------------------------------
# Transformer: IterativeSplitter
# ------------------------------
class IterativeSplitter(BaseEstimator, TransformerMixin):
    def __init__(self, text_column: str = "cleanedText", test_size: float = 0.2, random_state: int = 42):
        self.text_column = text_column
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X: Any, y: Any = None):
        return self

    def transform(self, X_labeltuple: Tuple[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """
        Expects a tuple: (df_with_labels_attached, label_matrix)
        Returns a dict with train_texts/test_texts and y_train/y_test (numpy arrays)
        """
        df, label_matrix = X_labeltuple
        texts = df[self.text_column].tolist()

        Xarr = np.array(texts).reshape(-1, 1)
        y = label_matrix

        X_train, y_train, X_test, y_test = iterative_train_test_split(
            Xarr, y, test_size=self.test_size
        )

        train_texts = [t[0] for t in X_train]
        test_texts = [t[0] for t in X_test]

        return {"train_texts": train_texts, "test_texts": test_texts, "y_train": y_train, "y_test": y_test}


# ------------------------------
# Transformer: Saver
# ------------------------------
class Saver(BaseEstimator, TransformerMixin):
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def fit(self, X: Any, y: Any = None):
        return self

    def transform(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        # data_dict should contain train_texts, test_texts, y_train, y_test and optionally 'mlb'
        train_texts = data_dict["train_texts"]
        test_texts = data_dict["test_texts"]
        y_train = data_dict["y_train"]
        y_test = data_dict["y_test"]
        mlb = data_dict.get("mlb", None)

        pd.Series(train_texts).to_csv(os.path.join(self.out_dir, "train_texts.csv"), index=False, header=False)
        pd.Series(test_texts).to_csv(os.path.join(self.out_dir, "test_texts.csv"), index=False, header=False)

        np.save(os.path.join(self.out_dir, "y_train.npy"), y_train)
        np.save(os.path.join(self.out_dir, "y_test.npy"), y_test)

        if mlb is not None:
            with open(os.path.join(self.out_dir, "mlb.pkl"), "wb") as f:
                pickle.dump(mlb, f)

        print(f"Saved processed files to {self.out_dir}")
        return data_dict


# ------------------------------
# Sklearn-like Pipeline wrapper
# ------------------------------
class SklearnLikePipeline:
    """
    A true sklearn-style pipeline that:
      - calls fit() on each transformer
      - then calls transform()
      - passes output of each stage to the next
    """
    def __init__(self, steps):
        self.steps = steps

    def run(self):
        data = None
        for name, transformer in self.steps:
            # Fit the transformer first
            transformer.fit(data)

            # Then transform
            data = transformer.transform(data)

        return data



# ------------------------------
# Example usage (main)
# ------------------------------
if __name__ == "__main__":
    RAW_DIR = "../datasets/emotion-classification/"
    OUT_DIR = "../processed/emotion-classification/"
    FILE_PREFIX = "emotion-recognition-"
    FILE_RANGE = list(range(1, 7))

    loader = DatasetLoader(raw_dir=RAW_DIR, prefix=FILE_PREFIX, file_range=FILE_RANGE)
    cleaner = TextCleaner(text_column="Text", out_column="cleanedText")
    emot_split = EmotionSplitter(emotion_column="Emotion", out_column="emotion_list", separator=None)
    mlb_transformer = MultiLabelEncoderTransformer(input_column="emotion_list")
    splitter = IterativeSplitter(text_column="cleanedText", test_size=0.2)
    saver = Saver(out_dir=OUT_DIR)

    pipeline = SklearnLikePipeline(steps=[
        ("loader", loader),
        ("cleaner", cleaner),
        ("emotion_splitter", emot_split),
        ("mlb", mlb_transformer),
        ("splitter", splitter),
        ("saver", saver)
    ])

    # Run pipeline
    result = pipeline.run()
    # 'result' is the dictionary returned by splitter -> saver returns same dict
