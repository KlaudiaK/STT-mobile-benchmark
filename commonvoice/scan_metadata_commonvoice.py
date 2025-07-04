import os
from pydub.utils import mediainfo
from tqdm import tqdm
import pandas as pd

from mutagen.mp3 import MP3
import os


def get_mp3_duration(file_path):
    audio = MP3(file_path)
    return audio.info.length


def scan_mp3_with_metadata(audio_dir, meta_df):
    data = []

    for file in tqdm(os.listdir(audio_dir)):
        if file.endswith(".mp3"):
            full_path = os.path.join(audio_dir, file)

            # try:
            # Get duration
            duration = get_mp3_duration(full_path)

            # Match metadata
            meta_row = meta_df[meta_df["path"] == file]
            if meta_row.empty:
                print(f"⚠️ Metadata not found for {file}")
                continue

            row = meta_row.iloc[0]
            data.append(
                {
                    "filename": file,
                    "client_id": row["client_id"],
                    "sentence": row["sentence"],
                    "age": row["age"],
                    "gender": row["gender"],
                    "locale": row["locale"],
                    "duration_sec": duration,
                }
            )

        #   except Exception as e:
        #       print(f"Error with {file}: {e}")

    return pd.DataFrame(data)


if __name__ == "__main__":
    print("[1/3] Scanning dataset...")
    df_meta = pd.read_csv("transcript_en_test.tsv", sep="\t")
    final_df = scan_mp3_with_metadata(audio_dir="en_test_0", meta_df=df_meta)
    final_df.to_csv("cv_test_audio_metadata.csv", index=False)
    print("✅ Metadata matched and saved.")
