<div align="center">
<h1>Automated Generation of Species-Specific Training Data from Large, Unlabeled Acoustic Datasets for Supervised Birdsong Isolation</h1>


This repository allows researchers to train source separation models to isolate specific species of bird from passive field recordings.

Below are the basic steps to train a new model and use pretrained models. For more technical details, please see our paper, "Birdsong Species Isolation Using Machine Learning." 

<img width="1389" alt="before and after separation image" src="https://github.com/JustinSasek/Birdsong-Source-Separation/blob/main/Source%20Separation%20Examples/beforeAfter.png?raw=true">
Golden-Cheeked Warbler birdcall spectrograms before and after source separation.
<br>
<br>

<audio controls>
  <source src="https://github.com/JustinSasek/Birdsong-Source-Separation/raw/main/Source%20Separation%20Examples/before0.wav" type="audio/wav">
  Your browser does not support the audio elements.
</audio><audio controls>
  <source src="https://github.com/JustinSasek/Birdsong-Source-Separation/raw/main/Source%20Separation%20Examples/after0.wav" type="audio/wav">
</audio>
<audio controls>
  <source src="https://github.com/JustinSasek/Birdsong-Source-Separation/raw/main/Source%20Separation%20Examples/before1.wav" type="audio/wav">
</audio><audio controls>
  <source src="https://github.com/JustinSasek/Birdsong-Source-Separation/raw/main/Source%20Separation%20Examples/after1.wav" type="audio/wav">
</audio>
<audio controls>
  <source src="https://github.com/JustinSasek/Birdsong-Source-Separation/raw/main/Source%20Separation%20Examples/before2.wav" type="audio/wav">
</audio><audio controls>
  <source src="https://github.com/JustinSasek/Birdsong-Source-Separation/raw/main/Source%20Separation%20Examples/after2.wav" type="audio/wav">
</audio>
<audio controls>
  <source src="https://github.com/JustinSasek/Birdsong-Source-Separation/raw/main/Source%20Separation%20Examples/before3.wav" type="audio/wav">
</audio><audio controls>
  <source src="https://github.com/JustinSasek/Birdsong-Source-Separation/raw/main/Source%20Separation%20Examples/after3.wav" type="audio/wav">
</audio>
<audio controls>
  <source src="https://github.com/JustinSasek/Birdsong-Source-Separation/raw/main/Source%20Separation%20Examples/before4.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio><audio controls>
  <source src="https://github.com/JustinSasek/Birdsong-Source-Separation/raw/main/Source%20Separation%20Examples/after4.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>

Golden-Cheeked Warbler birdcall audio before and after source separation
</div>



## Compute Requirements
For training new models, it is strongly recommended to have a CUDA-enabled GPU. 
For using pretrained models, a GPU is not required but recommended for faster processing. A large amount of RAM is also recommended. If the python processes crash, they likely ran out of RAM.

## Training
**Step 1**: Move your raw field recordings into `Data/Raw Recordings`. *Note: Recordings over 1 hour may cause memory issues.*

    Data
        Background Samples
        Positive Samples
        Raw Recordings
            raw1.wav
            raw2.wav
            ...
    README.md
    ...

**Step 2**: Edit `BIRD_OF_INTEREST` and `BIRD_OF_INTEREST_MIN_FREQ` within `globalVars.py` to match your species.

**Step 3**: Install libraries. *Note: This repo is designed for Python 3.10.9*

```bash
    pip install -r requirements.txt
```

**Step 4**: Extract and clean raw recordings:

```bash
    python3 dataCleaner.py
```
This will:
* Download the weights for the BirdNET and PANNs classifiers
* Run BirdNET and PANNs classifiers on raw recordings
* Extract <u>positive samples</u> (birdcalls of interest)
  * Clean birdcalls of interest with high-pass Butterworth filter, peak frequency normalization, and stationary spectral gating noise reduction
* Extract <u>label based</u> background audio by separating other birdcalls and audio categories
* Extract <u>volume based</u> background audio by taking the loudest minute from each raw recording (excluding birdcalls of interest) 

**Step 5**: Manually review data. Make sure positive samples are clean and audible. Remove noisy samples. Also, make sure no positive samples are in the background audio.

**Step 6**: Organize your birdcalls by song type and any rare song variations. Then modify the `DISTRIBUTION` hashmap in `globalVars.py` to match. This balances the dataset by using rarer birdcalls more often during training.

Ex:

    Data
        Positive Samples
            A Song
                With Hook
                Without Hook
            B Song
        Background Samples
            Volume Based
            Label Based
        Raw Recordings
    README.md
    ...

```python
DISTRIBUTION = {
    'Positive Samples/A Song/With Hook': 30,
    'Positive Samples/A Song/Without Hook': 10,
    'Positive Samples/B Song': 60,
    'Background Samples/Volume Based': 40,
    'Background Samples/Label Based': 60,
}
```

**Step 7**: Start training:

```bash
    python3 train.py
```

**Step 8**: Adjust parameters in `globalVars.py` if needed.

## Pretrained Models
We provide pretrained models for separating Golden-Cheeked Warbler birdsong from background noise. These models can be found in the `Checkpoints` folder.

To used the pretrained models, follow the steps below:

**Step 1**: Install libraries.

```bash
    pip install -r requirements.txt
```

**Step 2**: Use 'forward.py' to perform source separation:

Ex:

```python
from forward import Model
import soundfile as sf

model = Model(checkpointPath='Checkpoints/SUDO72-3sec.ckpt', modelType='SuDORMRFNet')
x, sr = sf.load('file.wav')
isolated, background = model.forward(x)
sf.write('isolated.wav', isolated, sr)


```

## Libraries Used

* Asteroid: https://github.com/asteroid-team/asteroid
* BirdNET: https://github.com/kahst/BirdNET-Analyzer
* PANNs: https://github.com/qiuqiangkong/audioset_tagging_cnn
* Microsoft Virtual Forest Impulse Response Simulator: https://github.com/microsoft/Forest_IR_synthesis
