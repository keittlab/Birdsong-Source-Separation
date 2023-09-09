from torch.utils import data
import os
import soundfile as sf
from scipy.signal import resample
import numpy as np
import torch
import librosa

# from main import DATA_PATH
from globalVars import GENERATED_READ_DATA_PATHS, SEGMENT_LENGTH, SR, JIT_GENRATION, TRAINING_PERCENTAGE, SAMPLES_PER_EPOCH
from dataGenerator import generateExample


# Dataset to load in audio files or generate them just in time
class BirdsongDataset(data.Dataset):
    dataset_name = "BirdsongDataset"

    # This function is called when the dataset is initialized
    def __init__(self, Val=False, **generatorVars):
        self.i = 0
        self.Val = Val
        self.generatorVars = generatorVars
        super(BirdsongDataset, self).__init__()

        if JIT_GENRATION:
            pass
        else:
            # get filepaths if not JIT
            if not os.path.exists(GENERATED_READ_DATA_PATHS + '/Training Data'):
                print('no data found')

            self.filepaths = []
            for path in GENERATED_READ_DATA_PATHS:
                for root, dirs, files in os.walk(path + '/Training Data/Isolated'):
                    for file in files:
                        filepath = os.path.join(root, file)
                        if filepath.endswith(".wav"):
                            self.filepaths.append(filepath)
            l = len(self.filepaths)
            self.trainLen = int(l * 0.8)
            self.valLen = l - self.trainLen

    # This function returns the length of the dataset

    def __len__(self):
        if JIT_GENRATION:
            if self.Val:
                return int(.2 * SAMPLES_PER_EPOCH)
            return int(.8 * SAMPLES_PER_EPOCH)

        if self.Val:
            return int(self.valLen * (60 // SEGMENT_LENGTH) * TRAINING_PERCENTAGE)
        return int(self.trainLen * (60 // SEGMENT_LENGTH) * TRAINING_PERCENTAGE)

    # This function returns the data for a given index

    def __getitem__(self, idx):
        isolated, background, combined = None, None, None
        if JIT_GENRATION:
            # generate audio just in time
            isolated, background, combined, _ = generateExample(length=SEGMENT_LENGTH * SR, **self.generatorVars)

            isolated = torch.from_numpy(isolated)
            background = torch.from_numpy(background)
            combined = torch.from_numpy(combined)
        else:
            start = (idx % (60 // SEGMENT_LENGTH)) * SEGMENT_LENGTH
            end = start + SEGMENT_LENGTH
            idx = idx // (60 // SEGMENT_LENGTH)
            if self.Val:
                idx += self.trainLen
            filepath = self.filepaths[idx]
            if not os.path.exists(filepath):
                raise Exception(filepath + 'not found while trying to load data')

            # Combined
            combFilepath = filepath.replace('Isolated', 'Combined').replace('isolated', 'combined')
            sr = sf.info(combFilepath).samplerate
            combined, _ = sf.read(combFilepath, dtype="float32", start=start * sr, stop=end * sr)
            combined = resample(combined, int(len(combined) / sr * SR))
            combined = torch.from_numpy(combined)

            sr = sf.info(filepath).samplerate
            isolated, _ = sf.read(filepath, dtype="float32", start=start * SR, stop=end * SR)
            isolated = resample(isolated, int(len(isolated) / sr * SR))

            backgroundFilepath = filepath.replace('Isolated', 'Background').replace('isolated', 'background')
            sr = sf.info(backgroundFilepath).samplerate
            background, _ = sf.read(backgroundFilepath, dtype="float32", start=start * SR, stop=end * SR)
            background = resample(background, int(len(background) / sr * SR))

        sources = np.vstack([isolated, background])
        # Convert sources to tensor
        sources = torch.from_numpy(sources)
        if self.i < 10 or self.i % 100 == 0:
            self.i += 1
        return combined, sources
