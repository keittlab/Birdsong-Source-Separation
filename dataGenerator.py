import os
from os.path import join
import librosa
import soundfile as sf
import numpy as np
import cv2
import multiprocessing as mp
from globalVars import RAW_DATA_PATH, GENERATED_WRITE_DATA_PATH, SR, PRELOAD_AUDIO, VIRTUAL_FOREST_IR_PATH, DISTRIBUTION
import concurrent.futures


# either read audio from or form dictionary
def loadAudio(file, audio):
    if PRELOAD_AUDIO:
        return audio[file]
    else:
        return librosa.load(file, sr=SR)[0]


# load all files into dictionary
def loadAllAudio():
    manager = mp.Manager()  # Shares dataset among dataloaders
    audio = manager.dict()
    for root, dirs, files in os.walk(RAW_DATA_PATH + '/Positive Samples'):
        for file in files:
            if file.endswith('.wav'):
                y, sr = librosa.load(join(root, file), sr=SR)
                if sr != SR:
                    print('Sample rate is not 22050', flush=True)
                    continue
                audio[join(root, file)] = y
    i = 0
    for root, dirs, files in os.walk(RAW_DATA_PATH + '/Background Samples'):
        for file in files:
            if file.endswith('.wav'):
                y, sr = librosa.load(join(root, file), sr=SR)
                if sr != SR:
                    print('Sample rate is not 22050', flush=True)
                    continue
                audio[join(root, file)] = y
                i += 1
    for root, dirs, files in os.walk(VIRTUAL_FOREST_IR_PATH):
        for file in files:
            if file.endswith('.wav'):
                y, sr = librosa.load(join(root, file), sr=SR)
                if sr != SR:
                    print('Sample rate is not 22050', flush=True)
                    continue
                audio[join(root, file)] = y
                i += 1
    if len(audio) == 0:
        raise Exception('No audio files found')
    print('\nfinished loading audio into memory\n')
    return audio


# saves filepaths and probabilities to lists
# also loads all files into memory if applicable
# reads from RAW_DATA_PATH
def initGenerator():
    if not os.path.exists(RAW_DATA_PATH + '/Positive Samples'):
        raise Exception('Positive Samples folder not found')
    if not os.path.exists(RAW_DATA_PATH + '/Background Samples'):
        raise Exception('Background Samples folder not found')
    
    audio = None
    if PRELOAD_AUDIO:
        audio = loadAllAudio()

    # load the files and their distribution
    pFiles = []
    pProbs = []
    
    nFiles = []
    nProbs = []
    
    for path, weight in DISTRIBUTION.items():
        # adds files to either positive or negative list
        if path.startswith('Positive Samples'):
            loadToList(pFiles, pProbs, weight, join(RAW_DATA_PATH, path))
        elif path.startswith('Background Samples'):
            loadToList(nFiles, nProbs, weight, join(RAW_DATA_PATH, path))
        else:
            raise Exception(f'unknown path found in distribution.csv: {path}')

    # Normalize the probabilities
    pProbs /= np.sum(pProbs)
    nProbs /= np.sum(nProbs)

    print('Generator Initialized', flush=True)
    
    return {'pFiles': pFiles, 'pProbs': pProbs, 'nFiles': nFiles, 'nProbs': nProbs, 'audio': audio}
    

# makes path if not found
def checkPath(path):
    if not os.path.exists(path):
        os.makedirs(path)


# concurrently generates training data
# writes to GENERATED_WRITE_DATA_PATH
def generateTrainingData(n = 125000):
    OFFSET = 0
    print('offset = %i' % OFFSET)
    
    initGenerator()
    
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=272) as executor:
        for i in range(n):
            if len(futures) > 272:
                for future in futures:
                    future.result()
                futures = []
            print('Generating Training Data - %i / %i' % (i + 1, n), flush=True)
            # Generates a 60s example with 60s of negative audio and 0 to 8 positive audio files
            j = i + OFFSET
            tenThousands = j // 10000 * 10000
            hundreds = j % 10000 // 100 * 100
            checkPath(GENERATED_WRITE_DATA_PATH + '/Training Data/Isolated/%i' % tenThousands)
            checkPath(GENERATED_WRITE_DATA_PATH + '/Training Data/Isolated/%i/%i' % (tenThousands, hundreds))
            checkPath(GENERATED_WRITE_DATA_PATH + '/Training Data/Combined/%i' % tenThousands)
            checkPath(GENERATED_WRITE_DATA_PATH + '/Training Data/Combined/%i/%i' % (tenThousands, hundreds))
            checkPath(GENERATED_WRITE_DATA_PATH + '/Training Data/Background/%i' % tenThousands)
            checkPath(GENERATED_WRITE_DATA_PATH + '/Training Data/Background/%i/%i' % (tenThousands, hundreds))
            
            future = executor.submit(genAndWriteExample, 
                            GENERATED_WRITE_DATA_PATH + '/Training Data/Isolated/%i/%i/isolated-%i.wav' % (tenThousands, hundreds, j), 
                            GENERATED_WRITE_DATA_PATH + '/Training Data/Combined/%i/%i/combined-%i.wav' % (tenThousands, hundreds, j),
                            GENERATED_WRITE_DATA_PATH + '/Training Data/Background/%i/%i/background-%i.wav' % (tenThousands, hundreds, j))
            futures.append(future)
            
    for future in futures:
        future.result()


# generates examples and also writes to disk
def genAndWriteExample(isoWritepath, combWritepath, backWritepath):
    isolated, background, combined, _ = generateExample()
    
    sf.write(isoWritepath, isolated, SR)
    sf.write(backWritepath, background, SR)
    sf.write(combWritepath, combined, SR)
    return


# randomly selects background audio and combines them
# also randomly selects volume
def loadBackground(length, nFiles, nProbs, audio):
    # load the negative files
    l = 0
    negativeCombined = np.array([], dtype = np.float32)

    while l < length:
        file = np.random.choice(nFiles, p=nProbs)
        y = loadAudio(file, audio)
        l += len(y)
        negativeCombined = np.append(negativeCombined, y)

    # pick random volume for negative audio
    negativeCombined *= np.random.choice([0.1, 0.5, 1], p=[0.05, 0.05, 0.9])

    
    start = np.random.randint(0, len(negativeCombined) - length + 1)
    end = start + length
    return negativeCombined[start:end]


# generates combined, isolated, and background audio
# set song to type if you only want to use a certain type of song
# set numPositiveFiles to a number if you want to use a specific number of positive files
# set IBR to override isolated-background ratio
# set doubleBackground to true if you want to double the background
# set forest to number of trees (in thousands) or a numpy array of the IR
def generateExample(length=60 * SR, songPath=None, numPositiveFiles=None, IBR=None, doubleBackground = False, forest=None, pFiles=None, pProbs=None, nFiles=None, nProbs=None, audio=None):
    # Filter based on songPath
    if songPath != None:
        goodIndicies = np.zeros(len(pProbs), dtype = np.bool)
        numGood = 0
        for i in range(len(pFiles) - 1, -1, -1):
            if songPath in pFiles[i]:
                goodIndicies[i] = True
                numGood += 1
        if numGood == 0:
            raise Exception(f'no files found in path {songPath} ')
        newPFiles = []
        newPProbs = np.empty(numGood, dtype = np.float32)
        iNew = 0
        for i in range(len(pFiles)):
            if goodIndicies[i]:
                newPFiles.append(pFiles[i])
                newPProbs[iNew] = pProbs[i]
                iNew += 1
        pFiles = newPFiles
        pProbs = newPProbs / np.sum(newPProbs)
    if len(pFiles) == 0:
        print(songPath)
        raise Exception('All positive files filtered out')
        
    # randomly select number of positive files
    if numPositiveFiles != None:
        pass
    elif length >= 6 * SR:
        numProbs = [20, 5, 5, 5, 5, 10, 15, 20, 15, 10, 5]
        numProbs /= np.sum(numProbs)
        numPositiveFiles = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], p=numProbs) # number of positive files to use
        numPositiveFiles *= length / (30 * SR) # scale the number of positive files to the length of the example
        numPositiveFiles = round(numPositiveFiles) # round to nearest integer
        numPositiveFiles = min(numPositiveFiles, len(pFiles))
    else:
        numPositiveFiles = np.random.choice([0,1], p=[.2, .8])
            
    # randomly select p positive files
    pFilesSample = np.random.choice(
        pFiles, size=numPositiveFiles, replace=False, p=pProbs)
    
    # load the positive files
    positiveWavs = []
    for file in pFilesSample:
        y = loadAndAugmentAudio(file, audio)
        positiveWavs.append(y)

    # load the negative files
    background = loadBackground(length, nFiles, nProbs, audio)
    if doubleBackground:
        background += loadBackground(length, nFiles, nProbs, audio)
        background /= 2
    
    isolated = layer(positiveWavs, length)
    
    # Convoled with virtual forest IR if applicable
    # if forest type is a np array convole with it
    if type(forest) == np.ndarray:
        isolated = np.convolve(isolated, forest)[:length]
    elif forest != None:
        if forest in [100, 200, 500, 1000]:
            ir = loadAudio(join(VIRTUAL_FOREST_IR_PATH, f'{forest}.wav'), audio)
            isolated = np.convolve(isolated, ir)[:length]
        else:
            raise Exception(f'unknown forest type \"{forest}\" found')
    
    # Adjust IBR ratio
    if IBR != None and numPositiveFiles > 0:
        sumIsolated = np.sum(np.abs(isolated))
        sumBackground = np.sum(np.abs(background))
        # Adjust IBR
        isolated = isolated * IBR * (sumIsolated + sumBackground) / (IBR+1) / sumIsolated
        background = background * (sumIsolated + sumBackground) / (IBR+1) / sumBackground
        
    IBR = np.sum(np.abs(isolated)) / np.sum(np.abs(background))
    
    combined = isolated + background
    
    return isolated, background, combined, numPositiveFiles, IBR

# gets list of all wave files in a directory
# Should recursively go through all subdirectories
# returns a list of sorted filepaths
def loadToList(list, probs, p, dir):
    init_len = len(list)
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith('.wav'):
                list.append(join(root, file))
    count = len(list) - len(probs)
    while len(probs) < len(list):
        probs.append(p / count)
    if len(list) - init_len == 0:
        raise Exception(f'No files found in {dir}')


# takes in a list of arrays which are wav files
# takes in the desired output array length
# randomly puts the arrays into the output array
# returns the output array
def layer(wavList, length):
    length = length + 2 * SR
    output = np.zeros(length, dtype=np.float32)  # add 2 seconds of padding to the beginning and end
    for wav in wavList:
        srcLen = len(wav)
        srcStart = 0
        if(srcLen > length):  # if the wav is longer than the output, randomly select a start point
            srcStart = np.random.randint(0, srcLen - length)
            srcLen = length
        destStart = np.random.randint(0, length - srcLen + 1)  # insertion point
        output[destStart:destStart + srcLen] += wav[srcStart:srcStart + srcLen]
    return output[1*SR: -1*SR]


# takes in a filepath to the wav file, an amount to shift the frequency axis, and a scale factor
# loads a wav file and converts it to a spectrogram
# bends the pitch by shifting the frequency axis
# Then, streches the spectrogram by a scale factor
# Then trims the spectrogram
# Then converts the spectrogram back to a wav file
# Returns as an array
def loadAndAugmentAudio(filepath, audio):
    # Shift from -15 to 15
    shiftAmount = int(-15 + 30 * np.random.rand())
    # Strech from 0.8 to 1.2
    strechScale = 0.8 + 0.4 * np.random.rand()
    # Volume from 0.5 to 1.5
    volumeMult = 0.15 + 1 * np.random.rand()
    # load the audio file
    x = loadAudio(filepath, audio)


    # convert to a spectrogram
    X = librosa.stft(x)

    # shift the frequency axis
    if shiftAmount > 0:
        X = np.pad(X, ((shiftAmount, 0), (0, 0)), 'constant')
        X = X[:-shiftAmount, :]
    elif shiftAmount < 0:
        X = np.pad(X, ((0, -shiftAmount), (0, 0)), 'constant')
        X = X[-shiftAmount:, :]

    # strech the spectrogram
    X = strech(X, strechScale)

    # trim the spectrogram
    X = X[:, 6:-6]

    # convert back to audio
    x = librosa.istft(X)
    return x * volumeMult


# takes in an spectograph as a 2d array
# stretchs the image by a scale factor on the x axis
def strech(X, scale):
    
    input_height, input_width = X.shape[:2]
    
    # Calculate the output width based on the scale factor
    new_width = int(input_width * scale)
    
    # Scale the input image horizontally
    X_real = np.float32(np.real(X))
    X_imag = np.float32(np.imag(X))
    X_real_new = cv2.resize(X_real, dsize=(new_width, input_height), interpolation = cv2.INTER_AREA)
    X_imag_new = cv2.resize(X_imag, dsize=(new_width, input_height), interpolation = cv2.INTER_AREA)
    return X_real_new + 1j * X_imag_new


if __name__ == "__main__":
    generateTrainingData(n = 1000000)
