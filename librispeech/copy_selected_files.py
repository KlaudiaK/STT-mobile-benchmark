import os
import shutil
import pandas as pd
import argparse


def process_subset(input_folder, subset_csv, output_folder, flat_structure=True):
    """Process subset with optional flat structure and single transcript file"""
    # Load subset
    subset = pd.read_csv(
        subset_csv,
        dtype={
            "filename": str,
            "speaker_id": str,
            "chapter_id": str,
            "sex": str,
            "duration_sec": float,
        },
    )

    os.makedirs(output_folder, exist_ok=True)
    transcriptions = []
    copied_files = 0

    for _, row in subset.iterrows():
        speaker_id = str(row["speaker_id"])
        chapter_id = str(row["chapter_id"])
        filename = str(row["filename"])
        base_name = os.path.splitext(filename)[0]

        # Source paths
        audio_src = os.path.join(input_folder, speaker_id, chapter_id, filename)
        trans_src = os.path.join(
            input_folder,
            speaker_id,
            chapter_id,
            f"{'-'.join(base_name.split('-')[:2])}.trans.txt",
        )

        # Destination handling
        if flat_structure:
            audio_dest = os.path.join(output_folder, filename)
        else:
            dest_dir = os.path.join(output_folder, speaker_id, chapter_id)
            os.makedirs(dest_dir, exist_ok=True)
            audio_dest = os.path.join(dest_dir, filename)

        try:
            # Copy audio file
            shutil.copy2(audio_src, audio_dest)

            # Extract transcription
            with open(trans_src, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith(f"{base_name} "):
                        transcriptions.append(line.strip())
                        break

            copied_files += 1

        except FileNotFoundError as e:
            print(f"⚠️ Missing file: {e.filename}")
        except Exception as e:
            print(f"⚠️ Error processing {filename}: {str(e)}")

    # Save SINGLE combined transcriptions file
    with open(
        os.path.join(output_folder, "combined_transcriptions.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write("\n".join(transcriptions))

    print(f"\n✅ Successfully processed {copied_files}/{len(subset)} files")
    print(f"Output folder: {os.path.abspath(output_folder)}")
    print(f"Structure: {'Flat' if flat_structure else 'Hierarchical'}")
    print(f"Transcripts saved to: combined_transcriptions.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="test-clean", help="Input folder path")
    parser.add_argument(
        "--subset",
        default="librispeech_test_clean_selected_subset.csv",
        help="Subset CSV file",
    )
    parser.add_argument(
        "--output",
        default="librispeech_test_clean_selected_audio",
        help="Output folder path",
    )
    parser.add_argument("--flat", action="store_true", help="Use flat output structure")
    args = parser.parse_args()

    print(f"Processing with {'flat' if args.flat else 'hierarchical'} structure...")
    process_subset(
        input_folder=args.input,
        subset_csv=args.subset,
        output_folder=args.output,
        flat_structure=args.flat,
    )
