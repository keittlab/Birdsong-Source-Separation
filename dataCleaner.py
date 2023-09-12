# imports
from globalVars import *
from util import *

import os
from os.path import join
import pandas as pd
import soundfile as sf
from scipy.signal import resample
import warnings
warnings.filterwarnings('ignore')

# Check for data
if not os.path.isdir(join(RAW_DATA_PATH, 'Raw Recordings')):
    raise Exception('Raw Recordings folder not found in Data folder')

# take cleanest 1500 rows from bird of interest above a certain threshold
# selects top 5 samples from every other column too above a certain threshold

df = pd.DataFrame()
df['file'] = ''
df['start time'] = 0
df['Flag'] = ''
df[BIRD_OF_INTEREST] = 0.0
df['Clean Prob'] = 0.0

fileIndex = 0
classifier = Classifier(pannsBatchSizeInMinutes=PANNS_BATCH_SIZE_IN_MINUTES)
for root, dirs, files in os.walk(join(RAW_DATA_PATH, 'Raw Recordings')):
    for file in files:
        if file.endswith('.wav'):
            print(f'\tprocessing {file}')
            audio_path = os.path.join(root, file)

            labels = classifier.classify(audio_path)
            print('\t\tfinished classification')
            if BIRD_OF_INTEREST not in labels.columns:
                labels[BIRD_OF_INTEREST] = 0.0

            # get loudest 1 minute of audio
            x, _ = librosa.load(audio_path, sr=SR)

            if len(x.shape) > 1:
                x = x[:, 0]
            y = getLoudestMinute(x, SR, labels, BIRD_OF_INTEREST, THRESHOLD)
            sf.write(join(RAW_DATA_PATH, f'Background Samples/Volume Based/{fileIndex}.wav'), y, SR)

            df = pd.concat((df, labels), ignore_index=True)
            if len(df) > 20000:  # limits memory usage
                df = consolidateLabels(df, BIRD_OF_INTEREST, THRESHOLD)
            fileIndex += 1


df = consolidateLabels(df, BIRD_OF_INTEREST, THRESHOLD)

print('\tsaving')
df = df.sort_values(by=['file', 'start time'], ascending=True)
df = df.reset_index(drop=True)
df.index += fileIndex
df.to_csv('Data/classifications.csv')

# Extract and clean audio

# iter rows and extract audio from files using filename and start time
# load with sf for processing
for index, row in df.iterrows():
    if index % 1000 == 0:
        print(f'\textracting files {index}-{min(index+1000,len(df))}')
    audio_path = row['file']
    start_time = max(int(row['start time']) - 1, 0)

    x, _ = librosa.load(audio_path, sr=SR, offset=start_time, duration=6)

    # convert to mono
    if len(x.shape) > 1:
        x = x[:, 0]

    if row['Flag'] == 'Positive':  # save to positive folder
        x = cleanAudio(x, SR, BIRD_OF_INTEREST_MIN_FREQ, NOISE_REDUCTION_ITERATIONS)
        write_path = join(RAW_DATA_PATH, f'Positive Samples/{index}.wav')
        sf.write(write_path, x, SR)

    elif row['Flag'] == 'Background':  # save to negative folder
        write_path = join(RAW_DATA_PATH, f'Background Samples/Label Based/{index}.wav')
        sf.write(write_path, x, SR)
