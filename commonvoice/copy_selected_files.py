import os
import shutil
import pandas as pd
import argparse


def process_commonvoice_subset(
    input_folder, subset_csv, output_folder, save_transcripts=True
):
    """Copy audio files listed in Common Voice subset CSV to output folder,
    optionally save combined transcripts file."""

    subset = pd.read_csv(subset_csv)
    os.makedirs(output_folder, exist_ok=True)

    transcriptions = []
    copied_files = 0

    for _, row in subset.iterrows():
        filename = row["filename"]
        audio_src = os.path.join(input_folder, filename)
        audio_dest = os.path.join(output_folder, filename)

        try:
            shutil.copy2(audio_src, audio_dest)
            copied_files += 1

            if save_transcripts:
                sentence = row.get("sentence", "")
                transcriptions.append(f"{filename}\t{sentence}")

        except FileNotFoundError:
            print(f"⚠️ Missing file: {audio_src}")
        except Exception as e:
            print(f"⚠️ Error copying {filename}: {str(e)}")

    if save_transcripts:
        transcript_path = os.path.join(output_folder, "combined_transcriptions.tsv")
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write("\n".join(transcriptions))
        print(f"Saved combined transcripts to {transcript_path}")

    print(f"\n✅ Successfully copied {copied_files}/{len(subset)} files")
    print(f"Output folder: {os.path.abspath(output_folder)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copy Common Voice subset files to a folder"
    )
    parser.add_argument(
        "--input", required=True, help="Input folder containing audio files"
    )
    parser.add_argument("--subset", required=True, help="CSV file with subset metadata")
    parser.add_argument(
        "--output", required=True, help="Output folder for copied files"
    )
    parser.add_argument(
        "--no-transcripts",
        action="store_true",
        help="Do not save combined transcripts file",
    )

    args = parser.parse_args()

    process_commonvoice_subset(
        input_folder=args.input,
        subset_csv=args.subset,
        output_folder=args.output,
        save_transcripts=not args.no_transcripts,
    )
