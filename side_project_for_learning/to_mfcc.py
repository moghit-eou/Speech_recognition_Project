import json
import os
import math
import librosa

DATASET_PATH = "./animals_sound"
JSON_PATH = "data_10.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 1  # 1 second long audio files
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512):
    """Extracts MFCCs from 1-second audio files and saves them into a JSON file along with genre labels."""

    # dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    # loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a genre sub-folder level
        if dirpath != dataset_path:  # Fix: Avoid adding the root dataset path
            # save genre label (i.e., sub-folder name) in the mapping
            genre_label = dirpath.split("/")[-1]
            data["mapping"].append(genre_label)
            print("\nProcessing: {}".format(genre_label))

            # process all audio files in genre sub-dir
            for f in filenames:
                # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                # extract mfcc from the whole 1-second signal
                mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                mfcc = mfcc.T  # Transpose to match the format

                # store the mfcc and label (genre index)
                data["mfcc"].append(mfcc.tolist())
                data["labels"].append(i)  # Store the genre label index
                print(f"{file_path}, MFCC extracted")

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH)
