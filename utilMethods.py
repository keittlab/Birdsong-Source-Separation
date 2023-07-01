import math
import numpy as np
import pandas as pd
import soundfile as sf
import librosa.display
import librosa
import os


# processes a long file
# removes bird calls identified by BirdNet
# trims the file to the n loudest seconds
def trimFromCSV(readFilepath, writeFilepath, csvpath, n):
    if not readFilepath.endswith(".wav"):
        return
    
    # loads in locations of bird calls
    df = pd.read_csv(csvpath)
    offsets = df[[df['filename'] == os.path.basename(readFilepath)]]['offset'].values
    offsets = np.sort(offsets)

    # load file
    x, sr = librosa.load(readFilepath, mono=True, sr=24000)
    initLen = len(x)
    x_new = []
    offsetIndex = 0
    i = 0
    
    # creates a new file without the bird calls
    while i < len(x):
        if offsetIndex < len(offsets) and i >= (offsets[offsetIndex]-10) * sr:
            i = (offsets[offsetIndex]+10) * sr
            offsetIndex += 1
            continue
        x_new.append(x[i])
        i += 1
    
    print(str((initLen - len(x_new)) // sr) + ' sec removed with csv')
    x_new = np.array(x_new)
    
    # trim to the n loudest seconds
    x_new = trimSilence(x_new, sr, n)
    
    # save file
    sf.write(writeFilepath, x_new, sr)


# Used sliding window to trim the n loudest seconds of the wav file
# Use this to get the most useful background audio from long files
def trimSilence(x, sr, n):
    # 1 second window
    halfWindowSize = sr // 2

    rollingSum = 0
    width = 0
    avgs = []
    
    # init rolling sum
    for i in range(0, halfWindowSize):
        rollingSum += abs(x[i])
        width += 1
        
    # fill with the original data if the average amplitude is above the threshold
    for i in range(0, len(x)):
        if i + halfWindowSize < len(x):
            rollingSum += abs(x[i + halfWindowSize])
            width += 1
        if i - halfWindowSize >= 0:
            rollingSum -= abs(x[i - halfWindowSize])
            width -= 1
        avgs.append(rollingSum / width)
        
    # determines the threshold for the n loudest seconds
    temp = np.array(avgs.copy())
    temp = np.sort(temp, 0)
    threshold = temp[max(len(avgs) - 1 - n*sr, 0)]
    
    x_new = []
    for i in range(0, len(avgs)):
        if avgs[i] > threshold:
            x_new.append(x[i])
    print('trimmed to ' + str(len(x_new) // sr) + ' sec')
    return x_new


# splits a wav file into multiple wav files each with their own bird call
# based on the amplitude of the audio in the window vs the surrounding window
def splitBirdsong(readFilepath, writeFilepath):
    x, sr = librosa.load(readFilepath, mono=True, sr=24000)
    length = len(x)

    minLength = 0.8 * sr  # in frames # minimum length of a bird call
    halfWindow = 25000  # in frames # half the size of the window
    threshold = 2.0  # in ratio of average in window to in surrounding window
    leftBuffer = math.floor(-1.5 * sr)  # in frames
    rightBuffer = math.floor(.75 * sr)  # in frames

    # Average in window
    avg = 0
    count = 0
    
    # Average in double the window size
    avgD = 0
    countD = 0
    
    # ID of file being created
    id = 0
    
    # Start pointer
    start = 0

    for i in range(halfWindow):
        avg += abs(x[i])
        count += 1
    for i in range(2*halfWindow):
        avgD += abs(x[i])
        countD += 1
    for i in range(length):
        if i - 2*halfWindow >= 0:
            avgD -= abs(x[i - 2*halfWindow])
            countD -= 1
        if i - halfWindow >= 0:
            avg -= abs(x[i - halfWindow])
            count -= 1
            
        if i + halfWindow < length:
            avg += abs(x[i + halfWindow])
            count += 1
        if i + 2*halfWindow < length:
            avgD += abs(x[i + 2*halfWindow])
            countD += 1
            
        if avg/count == 0:
            start = -1
            continue
        
        # ratio of average in window to in surrounding window
        if (avg/count) / ((avgD-avg)/(countD-count)) > threshold:
            if start == -1:
                start = i
        elif start == -1:
            continue
        elif i - start > minLength:
            # aggregate data
            data = []
            for j in range(max(start+leftBuffer, 0), min(i+rightBuffer, length)):
                data.append(x[j])
                
            # write to files
            path = writeFilepath.removesuffix(".wav") + '-' + str(id) + ".wav"
            sf.write(path, data, sr)
            id += 1
            start = -1
        else:
            start = -1

    print('splitting complete')


if __name__ == '__main__':
    trimFromCSV('readpath',
             'writepath',
             'csvpath', n=60*5)