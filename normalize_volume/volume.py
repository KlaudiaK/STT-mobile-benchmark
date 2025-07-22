import os
from pydub import AudioSegment
from pydub.playback import play
from pydub.effects import normalize

def normalize_audio_volume(input_folder, output_folder, target_dBFS=-20.0, file_ext="mp3"):
    """
    Normalizuje głośność wszystkich plików audio w folderze do określonego poziomu dBFS.
    
    :param input_folder: Ścieżka do folderu z plikami wejściowymi
    :param output_folder: Ścieżka do folderu, gdzie zostaną zapisane znormalizowane pliki
    :param target_dBFS: Docelowy poziom głośności w dBFS (domyślnie -20.0)
    :param file_ext: Rozszerzenie plików audio do przetworzenia (domyślnie "wav")
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(f".{file_ext}"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            print(f"Przetwarzanie pliku: {filename}")
            
            try:
                audio = AudioSegment.from_file(input_path, format=file_ext)
                normalized_audio = normalize(audio)
                
                change_in_dBFS = target_dBFS - normalized_audio.dBFS
                adjusted_audio = normalized_audio.apply_gain(change_in_dBFS)
         
                adjusted_audio.export(output_path, format=file_ext)
                print(f"Zapisano znormalizowany plik: {output_path}")
                
            except Exception as e:
                print(f"Błąd podczas przetwarzania pliku {filename}: {str(e)}")

if __name__ == "__main__":
    input_folder = "nagrania"  # Zmień na właściwą ścieżkę
    output_folder = "normalized"  # Zmień na właściwą ścieżkę
    target_volume = -20.0  # Docelowa głośność w dBFS (typowe wartości to między -20 a -16)
    
    normalize_audio_volume(input_folder, output_folder, target_volume)