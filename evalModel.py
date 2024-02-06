import os
import math
import numpy as np
from os.path import join
from datetime import datetime
import multiprocessing as mp
import decimal
from decimal import Decimal
decimal.getcontext().prec = 50

import torch
from scipy.optimize import fsolve

import soundfile as sf
import mir_eval

from globalVars import SR, DISTRIBUTION, BIRD_OF_INTEREST
from forward import Model
from dataGenerator import initGenerator, generateExample
from util import Classifier

from Virtual_Forest.ForestReverb import simulateForestIR


CSVPATH = 'Eval Results/Tests.csv'
WRITEPATH = None  #'Eval Results/Test Samples'  # uncomment to save some samples during testing
CHECKPOINT_PATH = f'Checkpoints/SUDO72-3sec.ckpt'
MODEL_TYPE = 'SuDORMRFNet'
TEST_SEGMENT_LENGTH = 60  # in seconds (can be longer than model segment length)

SPEED_OF_SOUND = 343.2  # in meters per second
csvLock = mp.Lock()
forwardLock = mp.Lock()  # inference is memory intensive


# how far ahead is the estimated signal from the actual signal in samples
def calcTimeLag(estimated, actual):
    if len(estimated) != len(actual):
        raise Exception('Lengths of estimated and actual signals must be equal')

    # Calculate cross-corre lation for isolated
    crossCorr = np.correlate(estimated, actual, mode='full')

    # Find the time lag corresponding to the maximum correlation
    timeLag = np.argmax(crossCorr) - len(estimated) + 1

    return timeLag


# find x,y coordinates of source given source separated signals from 3 mics
def trilaterate(input0, input1, input2):
    timeLag01 = calcTimeLag(input0, input1)
    timeLag02 = calcTimeLag(input0, input2)
    mic = [[0, 0], [10, 0], [0, 10]]

    d01 = timeLag01 / SR * SPEED_OF_SOUND
    d02 = timeLag02 / SR * SPEED_OF_SOUND

    def getTimeLagError(input):
        x = float(input[0])
        y = float(input[1])
        a = Decimal((x - mic[0][0])**2 + (y - mic[0][1])**2).sqrt()
        b = Decimal((x - mic[1][0])**2 + (y - mic[1][1])**2).sqrt()
        error01 = float(a) - float(b) - d01
        a = Decimal((x - mic[0][0])**2 + (y - mic[0][1])**2).sqrt()
        b = Decimal((x - mic[2][0])**2 + (y - mic[2][1])**2).sqrt()
        error02 = float(a) - float(b) - d02
        return [error01, error02]

    sol = fsolve(getTimeLagError, [0, 0])
    return sol


# Runs a single test - records the SNR, SIR, SAR and config to the csv file
def runSample(csv, i, config, generatorVars, isTimeLag=False):
    print(f'{datetime.now().strftime("%H:%M:%S")} starting eval {i} for {CHECKPOINT_PATH}', flush=True)
    newGeneratorVars = {**generatorVars, 'songPath': config[1], 'numPositiveFiles': config[2],
                        'IBR': config[3], 'doubleBackground': config[4], 'forest': config[5]}

    length = 6 * SR if isTimeLag else TEST_SEGMENT_LENGTH * SR

    isolated, background, combined, numPositiveFiles, IBR = generateExample(
        length=length, **newGeneratorVars)
    isolated = torch.from_numpy(isolated)
    background = torch.from_numpy(background)
    combined = torch.from_numpy(combined)
    sources = np.vstack([isolated, background])
    sources = torch.from_numpy(sources)

    line = f'{i}, {config[0]}, {config[1]}, {numPositiveFiles}, {IBR}, {config[4]}, {config[5]}, '

    forwardLock.acquire()
    global model
    try:
        estSources = model.forward(combined)
    except Exception as e:
        forwardLock.release()
        raise e
    forwardLock.release()

    if WRITEPATH != None and i < 20:
        sf.write(join(WRITEPATH, f'estIsolated{i}.wav'), estSources[0], SR)
        sf.write(join(WRITEPATH, f'estBackground{i}.wav'), estSources[1], SR)
        sf.write(join(WRITEPATH, f'combined{i}.wav'), combined, SR)

    sources = sources.detach().numpy()

    if isTimeLag:
        isolated = isolated.detach().numpy()
        timeLag = calcTimeLag(estSources[0], isolated) / SR
        line += f',,,,,,{timeLag}\n'
        if abs(timeLag) > 0:
            print('\t\texception found')
            sf.write(join(WRITEPATH, f'estIsolated{i}.wav'), estSources[0], SR)
            sf.write(join(WRITEPATH, f'estBackground{i}.wav'), estSources[1], SR)
            sf.write(join(WRITEPATH, f'combined{i}.wav'), combined, SR)
    else:
        sdr, sir, sar = None, None, None
        if np.all(sources[0] == 0):
            sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(
                sources[1], estSources[1], compute_permutation=False)
            line += f'0, 0, 0, {sdr[0]}, {sir[0]}, {sar[0]}\n'
        else:
            sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(sources, estSources, compute_permutation=False)
            line += f'{sdr[0]}, {sir[0]}, {sar[0]}, {sdr[1]}, {sir[1]}, {sar[1]}\n'

    csvLock.acquire()
    try:
        csv.write(line)
        csv.flush()
    except Exception as e:
        csvLock.release()
        raise e
    csvLock.release()


# Runs all tests n times - calls runSample
def runAll(n):
    generatorVars = initGenerator()
    i = 0
    csv = None
    if WRITEPATH != None and not os.path.exists(WRITEPATH):
        os.makedirs(WRITEPATH)

    # get last index
    if os.path.exists(CSVPATH):
        i = -1
        with open(CSVPATH, 'r') as csv:
            lines = csv.readlines()
            for j in range(1, len(lines)):
                index = int(lines[j].split(',')[0])
                i = max(index, i)
        i += 1
        csv = open(CSVPATH, 'a')  # open csv in append mode
    else:
        csv = open(CSVPATH, 'w')
        csv.write('Index, Test, Song Path, numPositiveFiles For 60 sec, IBR(Isolated Background Ratio), Double Background?, Forest (In thousands of trees), Iso SDR, Iso SIR, Iso SAR, Back SDR, Back SIR, Back SAR, Time Lag\n')
        csv.flush()

    global model
    model = Model(CHECKPOINT_PATH, MODEL_TYPE)

    runConfigs = [
        [5, 'Control', None, None, None, None, None],
        [1, 'Num Pos Songs', None, 0, None, None, None],
        [1, 'Num Pos Songs', None, 1, None, None, None],
        [1, 'Num Pos Songs', None, 2, None, None, None],
        [1, 'Num Pos Songs', None, 5, None, None, None],
        [1, 'Num Pos Songs', None, 10, None, None, None],
        [1, 'Num Pos Songs', None, 15, None, None, None],
        [1, 'Num Pos Songs', None, 20, None, None, None],
        [1, 'IBR', None, 20, 2, None, None],
        [1, 'IBR', None, 20, 1.5, None, None],
        [1, 'IBR', None, 20, 1, None, None],
        [1, 'IBR', None, 20, 0.5, None, None],
        [1, 'IBR', None, 20, 0.2, None, None],
        [1, 'IBR', None, 20, 0.1, None, None],
        [1, 'IBR', None, 20, 0.05, None, None],
        [1, 'IBR', None, 20, 0.02, None, None],
        [1, 'IBR', None, 20, 0.01, None, None],
        [3, 'Double Background', None, None, None, True, None],
        [1, 'Forest', None, None, None, None, 100],
        [1, 'Forest', None, None, None, None, 200],
        [1, 'Forest', None, None, None, None, 500],
        [1, 'Forest', None, None, None, None, 1000],
    ]
    
    for path in DISTRIBUTION:
        if path.startswith('Positive Samples'):
            runConfigs.append([2, 'Song', path, None, None, None, None])

    for _ in range(n):
        for config in runConfigs:
            for _ in range(config[0]):
                runSample(csv, i, config[1:], generatorVars)
                i += 1


# Runs control tests n times - calls runSample
def runControl(n):
    generatorVars = initGenerator()
    i = 0
    csv = None
    if WRITEPATH != None and not os.path.exists(WRITEPATH):
        os.makedirs(WRITEPATH)

    # get last index
    if os.path.exists(CSVPATH):
        i = -1
        with open(CSVPATH, 'r') as csv:
            lines = csv.readlines()
            for j in range(1, len(lines)):
                index = int(lines[j].split(',')[0])
                i = max(index, i)
        i += 1
        csv = open(CSVPATH, 'a')  # open csv in append mode
    else:
        os.makedirs(os.path.dirname(CSVPATH), exist_ok=True)
        csv = open(CSVPATH, 'w')
        csv.write('Index, Test, Song Path, numPositiveFiles For 60 sec, IBR(Isolated Background Ratio), Double Background?, Forest (In thousands of trees), Iso SDR, Iso SIR, Iso SAR, Back SDR, Back SIR, Back SAR, Time Lag\n')
        csv.flush()

    global model
    model = Model(CHECKPOINT_PATH, MODEL_TYPE)

    config = ['Control', None, None, None, None, None]
    for _ in range(n):
        runSample(csv, i, config, generatorVars)
        i += 1


# Tests that the separated source aligns with the original source - calls runSample
def runTimeLag(n):
    generatorVars = initGenerator()
    i = 0
    csv = None
    if WRITEPATH != None and not os.path.exists(WRITEPATH):
        os.makedirs(WRITEPATH)

    # get last index
    if os.path.exists(CSVPATH):
        i = -1
        with open(CSVPATH, 'r') as csv:
            lines = csv.readlines()
            for j in range(1, len(lines)):
                index = int(lines[j].split(',')[0])
                i = max(index, i)
        i += 1
        csv = open(CSVPATH, 'a')  # open csv in append mode
    else:
        csv = open(CSVPATH, 'w')
        csv.write('Index, Test, Song Path, numPositiveFiles For 60 sec, IBR(Isolated Background Ratio), Double Background?, Forest (In thousands of trees), Iso SDR, Iso SIR, Iso SAR, Back SDR, Back SIR, Back SAR, Time Lag\n')
        csv.flush()

    global model
    model = Model(CHECKPOINT_PATH, MODEL_TYPE)

    config = ['Time Lag', None, 1, None, None, None]
    for _ in range(n):
        runSample(csv, i, config, generatorVars, isTimeLag=True)
        i += 1


# Simulated trilateration environment and calculates error
def runTrilaterate(n):
    generatorVars = initGenerator()
    i = 0
    csv = None
    
    if not WRITEPATH is None and not os.path.isdir(WRITEPATH):
        os.makedirs(WRITEPATH)

    # get last index
    if os.path.exists(CSVPATH):
        i = -1
        with open(CSVPATH, 'r') as csv:
            lines = csv.readlines()
            for j in range(1, len(lines)):
                index = int(lines[j].split(',')[0])
                i = max(index, i)
        i += 1
        csv = open(CSVPATH, 'a')  # open csv in append mode
    else:
        csv = open(CSVPATH, 'w')
        csv.write('Index, x, y, IBR, est x before, est y before, est x after, est y after\n')
        csv.flush()

    global model
    model = Model()

    seeds = [np.random.randint(2**32 - 1) for _ in range(n * 10)]

    k = 0
    for _ in range(n):
        print(f'{datetime.now().strftime("%H:%M:%S")} starting eval {i} for {CSVPATH}', flush=True)
        x = np.random.rand() * 20 - 10
        y = np.random.rand() * 20 - 10
        line = f'{i},{x},{y},'
        
        # find the actual distance from the source to the closest mic
        dist0 = math.sqrt((x - 0)**2 + (y - 0)**2)
        dist1 = math.sqrt((x - 10)**2 + (y - 0)**2)
        dist2 = math.sqrt((x - 0)**2 + (y - 10)**2)
        actualDist = min(dist0, dist1, dist2)
        if actualDist > 10:
            continue
        
        irs = simulateForestIR(posSrc=np.array([x, y, 1.5]), micPoss=np.array(
            [[0, 0, 1.5], [10, 0, 1.5], [0, 10, 1.5]]))
        
        np.random.seed(seeds[i])
        _,_,comb0,_,ibr = generateExample(length=6 * SR, numPositiveFiles=1, forest=irs[:, 0], **generatorVars)
        np.random.seed(seeds[i])
        comb1 = generateExample(length=6 * SR, numPositiveFiles=1, forest=irs[:, 1], **generatorVars)[2]
        np.random.seed(seeds[i])
        comb2 = generateExample(length=6 * SR, numPositiveFiles=1, forest=irs[:, 2], **generatorVars)[2]
        iso0 = model.forward(comb0)[0]
        iso1 = model.forward(comb1)[0]
        iso2 = model.forward(comb2)[0]
        locationBefore = trilaterate(comb0, comb1, comb2)
        locationAfter = trilaterate(iso0, iso1, iso2)
        
        line += f'{ibr},{locationBefore[0]},{locationBefore[1]},{locationAfter[0]},{locationAfter[1]}\n'
        if k < 15 and math.sqrt((locationAfter[0] - x)**2 + (locationAfter[1] - y)**2) > 50:
            sf.write(join(WRITEPATH, f'iso0-{i}.wav'), iso0, SR)
            sf.write(join(WRITEPATH, f'iso1-{i}.wav'), iso1, SR)
            sf.write(join(WRITEPATH, f'iso2-{i}.wav'), iso2, SR)
            sf.write(join(WRITEPATH, f'comb0-{i}.wav'), comb0, SR)
            sf.write(join(WRITEPATH, f'comb1-{i}.wav'), comb1, SR)
            sf.write(join(WRITEPATH, f'comb2-{i}.wav'), comb2, SR)
            k += 1
        csv.write(line)
        i += 1


# Runs BirdNet
def classify(x, classifier):
    try:
        sf.write('temp.wav', x, SR)
        df = classifier.classify('temp.wav')
        if BIRD_OF_INTEREST not in df.columns:
            results = np.zeros(len(df))
        else:
            results = df[BIRD_OF_INTEREST].values # index 0: probability of bird from 0 to 3 seconds
    except KeyboardInterrupt:
        pass
    finally:
        os.remove('temp.wav')
    return results


# Tests BirdNet on separated source and combined source to measure improvement
def runBirdNet(n):
    generatorVars = initGenerator()
    i = 0
    csv = None
    
    if not WRITEPATH is None and not os.path.isdir(WRITEPATH):
        os.makedirs(WRITEPATH)

    # get last index
    if os.path.exists(CSVPATH):
        i = -1
        with open(CSVPATH, 'r') as csv:
            lines = csv.readlines()
            for j in range(1, len(lines)):
                index = int(lines[j].split(',')[0])
                i = max(index, i)
        i += 1
        csv = open(CSVPATH, 'a')  # open csv in append mode
    else:
        csv = open(CSVPATH, 'w')
        csv.write('Index, Type, IBR, Num Calls, Time 0\n')
        csv.flush()

    global model
    model = Model()
    classifier = Classifier(birdNetOnly = True)

    for j in range(n):
        # tests for various isolated background ratios
        for ibr in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 1.5, 2]:
            print(f'{datetime.now().strftime("%H:%M:%S")} starting eval {i} for {CSVPATH}', flush=True)
            iso, back, comb, numPositiveFiles, _ = generateExample(length=6 * SR, numPositiveFiles=j%2, IBR=ibr, **generatorVars)
            estIso = model.forward(comb)[0]
            if WRITEPATH != None and i < 20:
                sf.write(join(WRITEPATH, f'iso{i}.wav'), iso, SR)
                sf.write(join(WRITEPATH, f'comb{i}.wav'), comb, SR)
                sf.write(join(WRITEPATH, f'estiso{i}.wav'), estIso, SR)
            pComb = classify(comb, classifier)
            pEstIso = classify(estIso, classifier)
            csv.write(f'{i},Comb,{ibr},{numPositiveFiles},{",".join(str(p) for p in pComb)}\n')
            csv.write(f'{i},EstIso,{ibr},{numPositiveFiles},{",".join(str(p) for p in pEstIso)}\n')
            if i % 100 == 0 or j < 10:
                csv.flush()
            i += 1


if __name__ == '__main__':
    # Control Tests 
    runControl(10000)
    
    # Test BirdNet Improvement
    runBirdNet(10000)

    # Trilateration Testing
    runTrilaterate(10000)
        
    # All Other Tests: random forest, double background, IBR, num positive files, song path
    runAll(1000000)