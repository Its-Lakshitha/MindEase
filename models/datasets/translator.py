import time
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from deep_translator import GoogleTranslator

dataset_path = "emotion-recognition-dataset/data/train-00000-of-00001.parquet"
outputFile = "student_mh_counseling_100k.csv"
sourceLang = "vi"
targetLang = "en"

# Load dataset
if dataset_path.endswith(".csv"):
    df = pd.read_csv(dataset_path)
elif dataset_path.endswith(".parquet"):
    df = pd.read_parquet(dataset_path)
else:
    raise ValueError("Unsupported file format.")

# Demo: use first 1000 rows
sampled_df = df.iloc[:100000].copy()

# ------------------------------
# Translation functions
# ------------------------------

def translate_row(text):
    try:
        return GoogleTranslator(source=sourceLang, target=targetLang).translate(text)
    except:
        return text

def translate_chunk(df_chunk):
    """
    Runs inside top-level worker.
    Creates its own ThreadPool for the chunk (100 rows).
    """
    for col in df_chunk.select_dtypes(include=['object']).columns:
        with ThreadPoolExecutor(max_workers=20) as pool:  # inner parallelism
            translated = list(pool.map(translate_row, df_chunk[col].tolist()))
        df_chunk[col] = translated

    return df_chunk

# ------------------------------
# Split into 10 chunks of 100 rows
# ------------------------------

num_outer_threads = 10
chunks = np.array_split(sampled_df, num_outer_threads)

# ------------------------------
# Run the 10 top-level threads
# ------------------------------

start = time.time()
print(f"Starting translation: {start:.2f} seconds : Time: {time.ctime()}")

with ThreadPoolExecutor(max_workers=num_outer_threads) as outer_pool:
    processed_chunks = list(outer_pool.map(translate_chunk, chunks))

# Combine results
translated_df = pd.concat(processed_chunks, ignore_index=True)

end = time.time()

print(f"Total time: {end-start:.2f} seconds")

translated_df.to_csv(outputFile, index=False)
