import os
import pandas as pd
from pydub.utils import mediainfo
from tqdm import tqdm
import pandas as pd
from mutagen.mp3 import MP3


def get_mp3_duration(file_path):
    audio = MP3(file_path)
    return audio.info.length


def scan_commonvoice(audio_dir, metadata_path):
    """Scans Common Voice .mp3 files and enriches with duration"""
    df = pd.read_csv(metadata_path, sep="\t")

    data = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scanning audio"):
        filename = row["path"]
        full_path = os.path.join(audio_dir, filename)

        if not os.path.exists(full_path):
            print(f"Missing file: {filename}")
            continue

        try:
            duration = get_mp3_duration(full_path)
            data.append(
                {
                    "filename": filename,
                    "client_id": row.get("client_id", ""),
                    "sentence": row.get("sentence", ""),
                    "age": row.get("age", ""),
                    "gender": row.get("gender", ""),
                    "locale": row.get("locale", ""),
                    "duration_sec": duration,
                }
            )
        except Exception as e:
            print(f"Error with {filename}: {e}")

    return pd.DataFrame(data)


def select_balanced_subset(df, target_minutes=30):
    """Selects a balanced gender/duration subset"""
    df = df.copy()
    df = df[
        df["gender"].isin(["male_masculine", "female_feminine"])
    ]  # use only labeled genders

    df["duration_min"] = df["duration_sec"] / 60
    df["duration_cat"] = pd.cut(
        df["duration_sec"],
        bins=[0, 5, 10, float("inf")],
        labels=["short", "medium", "long"],
    )

    # Distribution weights
    group_weights = {
        ("female_feminine", "short"): 0.15,
        ("female_feminine", "medium"): 0.35,
        ("female_feminine", "long"): 0.15,
        ("male_masculine", "short"): 0.15,
        ("male_masculine", "medium"): 0.35,
        ("male_masculine", "long"): 0.15,
    }

    stratified = []
    for (gender, dur_cat), weight in group_weights.items():
        group = df[(df["gender"] == gender) & (df["duration_cat"] == dur_cat)]
        sample_size = max(1, int(len(group) * weight))
        if not group.empty:
            stratified.append(
                group.sample(min(sample_size, len(group)), random_state=42)
            )

    subset = pd.concat(stratified).sample(frac=1, random_state=42)
    subset = subset.iloc[: int(target_minutes / subset["duration_min"].mean())]

    return subset


if __name__ == "__main__":
    AUDIO_DIR = "en_test_0"  # Folder with .mp3 files
    METADATA_PATH = "transcript_en_test.tsv"

    print("[1/3] Scanning Common Voice audio directory...")
    df = scan_commonvoice(AUDIO_DIR, METADATA_PATH)
    df.to_csv("commonvoice_with_durations.csv", index=False)
    print("Saved full metadata to 'commonvoice_with_durations.csv'")

    print("\n[2/3] Selecting balanced subset...")
    subset = select_balanced_subset(df, target_minutes=30)

    print("\n=== Subset Summary ===")
    print(f"Total clips: {len(subset)}")
    print(f"Total duration: {subset['duration_min'].sum().round(2)} minutes")
    print("\nGender distribution:")
    print(subset["gender"].value_counts())
    print("\nDuration categories:")
    print(subset["duration_cat"].value_counts())

    subset.drop(columns=["duration_min"]).to_csv(
        "commonvoice_selected_subset.csv", index=False
    )
    print("\n[3/3] Saved balanced subset to 'commonvoice_selected_subset.csv'")
