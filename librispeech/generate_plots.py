import os
import pandas as pd
import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def plot_duration_distribution(df, save_path=None, show=True, set_name=None):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(df["duration_sec"], bins=50, kde=False, color="skyblue")
    plt.title(f"Rozkład długości plików audio w zbiorze {set_name}")
    plt.xlabel("Czas trwania w sekundach")
    plt.ylabel("Liczba plików")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def plot_gender_distribution(df, save_path=None, show=True, set_name=None):
    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 5))
    colors = ["#4C72B0", "#FF9203"]
    sns.countplot(data=df, x="sex", palette=colors, order=["F", "M", "U"])
    plt.title(f"Rozkład liczby nagrań wg płci mówcy w zbiorze {set_name}")
    plt.xlabel("Płeć mówcy", labelpad=15)
    plt.ylabel("Liczba nagrań")
    plt.xticks(ticks=[0, 1, 2], labels=["Kobieta (F)", "Mężczyzna (M)", "Nieznana (U)"])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def load_speaker_genders(speakers_file="SPEAKERS.TXT"):
    speaker_genders = {}
    with open(speakers_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith(";") and "|" in line:
                parts = [x.strip() for x in line.split("|")]
                speaker_id, sex = parts[0], parts[1]
                speaker_genders[speaker_id] = sex
    return speaker_genders


def scan_librispeech(root_dir="test-clean", speakers_file="SPEAKERS.TXT"):
    """Scans LibriSpeech directory and returns DataFrame with metadata"""
    speaker_genders = load_speaker_genders(speakers_file)
    data = []

    for root, _, files in tqdm(os.walk(root_dir), desc="Scanning folders"):
        for file in files:
            if file.endswith(".flac"):
                parts = root.split(os.sep)
                speaker_id, chapter_id = parts[-2], parts[-1]
                full_path = os.path.join(root, file)

                try:
                    duration = librosa.get_duration(filename=full_path)
                    data.append(
                        {
                            "filename": file,
                            "speaker_id": speaker_id,
                            "chapter_id": chapter_id,
                            "sex": speaker_genders.get(speaker_id, "U"),
                            "duration_sec": duration,
                        }
                    )
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")

    return pd.DataFrame(data)


def plot_combined_distribution(df, save_path=None, show=True):
    """
    Tworzy dwa wykresy obok siebie: rozkład długości plików audio i rozkład płci mówcy.

    Parametry:
    - df: DataFrame zawierający kolumny 'duration_sec' i 'sex'
    - save_path: ścieżka do zapisu wykresu (np. 'combined_distribution.png') lub None
    - show: czy wyświetlić wykres po utworzeniu
    """
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram długości
    sns.histplot(df["duration_sec"], bins=50, kde=False, color="skyblue", ax=axes[0])
    axes[0].set_title("Rozkład długości plików audio")
    axes[0].set_xlabel("Czas trwania (s)", labelpad=15)
    axes[0].set_ylabel("Liczba plików")

    # Rozkład płci
    sns.countplot(data=df, x="sex", palette="pastel", order=["F", "M", "U"], ax=axes[1])
    axes[1].set_title("Rozkład liczby nagrań wg płci mówcy")
    axes[1].set_xlabel("Płeć mówcy", labelpad=15)
    axes[1].set_ylabel("Liczba nagrań")
    axes[1].set_xticks([0, 1, 2])
    axes[1].set_xticklabels(["Kobieta (F)", "Mężczyzna (M)", "Nieznana (U)"])
    # fig.suptitle("Statystyki zbioru LibriSpeech test-clean", fontsize=16, y=1.02)

    fig.tight_layout()

    # fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    df = scan_librispeech("test-clean", "SPEAKERS.TXT")
    # df.to_csv("librispeech_with_gender.csv", index=False)
    print("Saved full metadata to 'librispeech_with_gender.csv'")
    # plot_duration_distribution(
    #     df,
    #     save_path="librispeech_test_clean_duration_distribution.png",
    #     set_name="Librispeech test-clean",
    # )
    # plot_gender_distribution(
    #     df,
    #     save_path="librispeech_test_clean_gender_distribution.png",
    #     set_name="Librispeech test-clean",
    # )
    plot_combined_distribution(df, save_path="librispeech_test_clean_distribution.png")
