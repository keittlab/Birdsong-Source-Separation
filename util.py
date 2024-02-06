import os
import pandas as pd
import numpy as np
import requests
import math
import shutil

import librosa
import soundfile as sf
import scipy
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from pathlib import Path
from panns_inference import SoundEventDetection, labels
import noisereduce as nr
from scipy import signal


def _getRollingSum(x, window):
    x_padded = np.pad(x, (math.ceil(window/2), math.floor(window/2)), 'constant', constant_values=(0, 0))
    cumsum_x = np.cumsum(x_padded)
    rolled_cumsum_x = cumsum_x[window:] - cumsum_x[:-window]
    return rolled_cumsum_x


# Labels audio with bird call and other sound labels
# Takes in audio path and returns pandas dataframe with labels
# If birdNetOnly is true, only returns birdNet labels
# pannsBatchSizeInMinutes defines how many minutes of audio to run PANNS on at a time
class Classifier:
    def __init__(self, birdNetOnly = False, pannsBatchSizeInMinutes = 30):
        self.birdNetOnly = birdNetOnly
        self.pannsBatchSizeInMinutes = pannsBatchSizeInMinutes
        
        # Downloads file from url and saves to save_path
        def download_file(url, save_path):
                    response = requests.get(url)
                    if response.status_code == 200:
                        with open(save_path, 'wb') as file:
                            file.write(response.content)
                        print(f"\t\tFile downloaded and saved as {save_path}")
                    else:
                        print(f"\t\tFailed to download file. Status code: {response.status_code}")
                        
        # init BirdNET
        self.analyzer = Analyzer()
        
        if not self.birdNetOnly:
            # init PANNs
            # PANNs model weights
            MODEL_URL = 'https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1'
            MODEL_PATH = 'Cnn14_DecisionLevelMax_mAP=0.385.pth'

            if os.path.isfile(MODEL_PATH):
                print(f'Model weights found for PANNs at {MODEL_PATH}')
            else:
                print('Downloading PANNs model weights (312Mb)')
                
                # Download weights
                download_file(MODEL_URL, MODEL_PATH)

            # Move labels file to required location for package
            CURR_LABELS_PATH = 'class_labels_indices.csv'
            REQUIRED_LABELS_PATH = '{}/panns_data/class_labels_indices.csv'.format(str(Path.home()))
            shutil.copy(CURR_LABELS_PATH, REQUIRED_LABELS_PATH)
            self.sed = SoundEventDetection(checkpoint_path=MODEL_PATH, device='cuda')

    def classify(self, audio_path):
        # Run BirdNET
        recording = Recording(
            self.analyzer,
            audio_path,
            min_conf=0,
            overlap=2.0,  # using 3-sec window that slides by 1 second
        )
        recording.analyze()
        df = pd.DataFrame(recording.detections)  # convert to pandas dataframe and compress
        df = df.apply(pd.to_numeric, errors='coerce')
        birdnetDf = df.pivot_table('confidence', index='start_time', columns='common_name',
                                   aggfunc='mean')  # pivot by common name
        birdnetDf = birdnetDf.fillna(0)
        
        if self.birdNetOnly:
            joined = birdnetDf
        else:
            # Run PANNS
            # Load audio
            audio, _ = librosa.core.load(audio_path, sr=32000, mono=True)
            initLength = audio.shape[0]
            initLengthinSecs = math.ceil(initLength / 32000)
            
            # Make batches
            batchSize = self.pannsBatchSizeInMinutes * 60 * 32000
            numBatches = math.ceil(initLength / batchSize)
            padding = numBatches * batchSize - initLength
            audio = np.pad(audio, (0, padding), mode='constant', constant_values=(0, 0))
            audio = np.reshape(audio, (numBatches, batchSize))

            # Run detection batch by batch
            firstOutput = self.sed.inference(audio[0:1, :])  # returns shape: (numBatches, 100*lengthInSecsPerBatch, n_classes)
            framewiseOutput = np.empty((audio.shape[0], firstOutput.shape[1], firstOutput.shape[2]), )
            framewiseOutput[0:1, :, :] = firstOutput
            for i in range(1, audio.shape[0]):
                framewiseOutput[i:i+1, :, :] = self.sed.inference(audio[i:i+1, :])
                
            framewiseOutput = np.reshape(framewiseOutput, (1, -1, framewiseOutput.shape[2]))
            framewiseOutput = framewiseOutput[:, :initLengthinSecs*100, :]


            # Pad frames to multiple of 100
            lengthInSecs = math.ceil(framewiseOutput.shape[1] / 100)
            framesNeeded = lengthInSecs * 100 - framewiseOutput.shape[1]
            framewiseOutput = np.pad(framewiseOutput, ((0, 0), (0, framesNeeded), (0, 0)),
                                    mode='constant', constant_values=0)

            # max pooling to second
            framewiseOutput = np.reshape(framewiseOutput, (-1, 100, 527)).max(axis=1)

            # Convert to pandas dataframe and compress
            pannsDf = pd.DataFrame(framewiseOutput, columns=labels)
            pannsDf = pannsDf.apply(pd.to_numeric, errors='ignore')

            # Remove conflicting labels
            badHeaders = ['Bird', 'Bird vocalization, bird call, bird song', 'Chirp, tweet', 'Squawk', 'Chirp tone',
                        'Bird flight, flapping wings', 'Pigeon, dove', 'Coo', 'Crow', 'Caw', 'Owl', 'Hoot']
            for badHeader in badHeaders:
                if badHeader in pannsDf.columns:
                    pannsDf = pannsDf.drop(badHeader, axis=1)

            # find rolling max for 6 seconds aligned with 3 second BirdNet window
            # we want the birdcall to be centered (1 sec before and 2 after works the best for this)
            # but we want the whole window to not have any background noise

            #            0     1     2     3     4     5
            # BirdNet          *     *     *
            # PANNs      *     *     *     *     *     *
            
            for col in pannsDf.columns:
                series = pannsDf[col]
                series = pd.Series([0] + series.tolist() + [0] * 2)  # pad with 0s
                pannsDf[col] = series.rolling(window=6).max().shift(-5)  # take rolling max and center as seen above

            # drop last 3 rows (from padded 0s)
            pannsDf = pannsDf.iloc[:-2]
        
            # join pannsDf with pivot on index value
            joined = birdnetDf.join(pannsDf, how='right', rsuffix='_panns')
        
        joined['file'] = audio_path
        joined['start time'] = joined.index
        
        return joined


# remove all audio that is potentially the bird of interest
def getLoudestMinute(x, sr, labels, birdOfInterest, threshold):
    
    def getRollingMax(x, window):
        left_pad = math.floor((window-1)/2)
        right_pad = math.ceil((window-1)/2)
        x_padded = np.pad(x, (left_pad, right_pad), 'constant', constant_values=(0, 0))
        rollingMax = pd.Series(x_padded).rolling(window).max().values[window-1:]
        return rollingMax
        
    
    ceilSeconds = math.ceil(len(x)/sr)
    keep = np.ones(ceilSeconds)
    
    # remove all audio that is potentially the bird of interest
    for _, row in labels.loc[labels[birdOfInterest] >= threshold].iterrows():
        start = int(row['start time'])  # 1 second before
        end = int(start + 6)  # 2 seconds after
        keep[start:end] = 0
    
    keepMask = np.repeat(keep, sr)[:len(x)]
    
    x = x[(keepMask == 1)]
    
    # get sliding sum with 1 second window or x
    rollingSum = _getRollingSum(np.abs(x), window=sr)
    
    # get rolling max of rollingSum
    rollingMax = getRollingMax(rollingSum, window=sr)
    
    threshold = np.sort(rollingMax)[-60 * sr]
    
    # trim to  loudest minute
    y = x[(rollingMax > threshold)]
    
    return y
    

# Consolidates labels from all audio files down to labels of interest
def consolidateLabels(df, birdOfInterest, threshold):
    df = df.fillna(0)
    
    df['Flag'] = 'Remove'
    
    # mark top 5 rows for every other column as Background
    special_cols = set(['Flag', 'Clean Prob', birdOfInterest, 'file', 'start time'])
    gen_cols = set(df.columns) - special_cols
    
    for col in gen_cols:
        colAndNoBirdOfInterest = df[col] * (1-df[birdOfInterest])
        
        # remove overlap by only keeping local maxima
        colMaxima = colAndNoBirdOfInterest * (colAndNoBirdOfInterest == colAndNoBirdOfInterest.rolling(6, center=True).max())
        
        # top 5 rows for every other column
        top5_index = set(colMaxima.nlargest(5).index)
        df.loc[(colAndNoBirdOfInterest > threshold) & (df.index.isin(top5_index)), 'Flag'] = 'Background'
        
    # top 1500 rows for bird of interest
    df['Clean Prob'] = df[birdOfInterest] * (1 - df.loc[:, gen_cols]).prod(axis=1)
    
    # remove overlap by only keeping local maxima
    cleanProbMaxima = df['Clean Prob'] * (df['Clean Prob'] == df['Clean Prob'].rolling(6, center=True).max())
        
    # mark top 1500 rows as Positive
    top1500_index = set(cleanProbMaxima.nlargest(1500).index)
    df.loc[(df['Clean Prob'] > threshold) & (df.index.isin(top1500_index)), 'Flag'] = 'Positive'
        
    df = df.loc[df['Flag'] != 'Remove']
    df = df.reset_index(drop=True)
    return df


# Performs high-pass filter, spectral gating noise reduction, and normalization
def cleanAudio(x, sr, lowerCutoffFrequency, noiseReductionIterations):
    # high-pass Butterworth filter
    nyquist = 0.5 * sr
    high_pass_order = 4
    b, a = signal.butter(high_pass_order, lowerCutoffFrequency / nyquist, btype='high', analog=False)
    x = signal.filtfilt(b, a, x)
    
    # reduce peak frequency of audio to -1 dB
    peak_amplitude = np.max(np.abs(x))
    desired_amplitude = 10 ** (-1 / 20)
    x = x * (desired_amplitude / peak_amplitude)
    
    # Perform stationary spectral gating noise reduction
    for _ in range(noiseReductionIterations):
        x = nr.reduce_noise(y=x, sr=sr, y_noise=x, stationary=True) # Apply noise reduction
    
    return x
