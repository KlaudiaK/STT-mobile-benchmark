import os
import pandas as pd
import librosa
from tqdm import tqdm

def load_speaker_genders(speakers_file="SPEAKERS.TXT"):
    """Loads speaker genders from SPEAKERS.TXT"""
    speaker_genders = {}
    with open(speakers_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.startswith(';') and '|' in line:
                parts = [x.strip() for x in line.split('|')]
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
                    data.append({
                        "filename": file,
                        "speaker_id": speaker_id,
                        "chapter_id": chapter_id,
                        "sex": speaker_genders.get(speaker_id, "U"),
                        "duration_sec": duration,
                    })
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")
    
    return pd.DataFrame(data)

def select_balanced_subset(df, target_minutes=120):
    """Selects balanced subset by gender and duration"""
    df['duration_min'] = df['duration_sec'] / 60
    df['duration_cat'] = pd.cut(
        df['duration_sec'],
        bins=[0, 5, 10, float('inf')],
        labels=['short', 'medium', 'long']
    )
    
    # Calculate sample sizes per group
    group_weights = {
        ('F', 'short'): 0.15,
        ('F', 'medium'): 0.35,
        ('F', 'long'): 0.15,
        ('M', 'short'): 0.15,
        ('M', 'medium'): 0.35,
        ('M', 'long'): 0.15,
    }
    
    stratified = []
    for (gender, dur_cat), weight in group_weights.items():
        group = df[(df['sex'] == gender) & (df['duration_cat'] == dur_cat)]
        sample_size = max(1, int(len(group) * weight))
        stratified.append(group.sample(sample_size, random_state=42))
    
    subset = pd.concat(stratified).sample(frac=1, random_state=42)  # Shuffle
    subset = subset.iloc[:int(target_minutes / subset['duration_min'].mean())]  # Approximate target
    
    return  subset

if __name__ == "__main__":
    # Step 1: Scan and save full metadata
    print("[1/3] Scanning dataset...")
    df = scan_librispeech("test-clean", "SPEAKERS.TXT")
    df.to_csv("librispeech_with_gender.csv", index=False)
    print("Saved full metadata to 'librispeech_with_gender.csv'")
    
  # Step 2: Select balanced subset
    print("\n[2/3] Selecting balanced subset...")
    subset = select_balanced_subset(df, target_minutes=120)
    
    # Print summary
    print("\n=== Subset Summary ===")
    print(f"Total clips: {len(subset)}")
    print(f"Total duration: {subset['duration_min'].sum().round(2)} minutes")
    print("\nGender distribution:")
    print(subset['sex'].value_counts())
    print("\nDuration categories:")
    print(subset['duration_cat'].value_counts())

    # Step 3: Save subset
    subset.drop(columns=['duration_min']).to_csv("selected_subset.csv", index=False)
    print("\n[3/3] Saved balanced subset to 'selected_subset.csv'")
  