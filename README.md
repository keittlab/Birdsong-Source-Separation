<div align="center">

**Birdsong Species Isolation Using Machine Learning**

</div>

--------------------------------------------------------------------------------


This repository contains the code for the paper [Birdsong Species Isolation Using Machine Learning].
It is built off of the Asteroid repository, which is a PyTorch-based audio source separation toolkit for researchers (https://github.com/asteroid-team/asteroid). This repository can be used to train your own birdsong source separation model or use the provided pretrained model to separate Golden-Cheeked Warbler from background noise.

## Training
Step 1: Create a folder called Data and organize your data within

    asteroid
        Data
            Background Samples
            Positive Samples
        main.py
        README.md
        ...

Step 2: Modify the dataset under 'asteroid/data/birdsong_dataset.py' to match your data

Step 3: Install libraries

```bash
    pip install -r requirements.txt
```

Step 4: Start training

```bash
    python3 main.py
```

Step 5: Adjust parameters in globalVars.py

## Pretrained models
We provide pretrained models for separating Golden-Cheeked Warbler birdsong from background noise. These models can be found in the `Checkpoints` folder.

To used the pretrained models, follow the steps below:

Step 1: Install libraries

```bash
    pip install -r requirements.txt
```

Step 2: Use 'forward.py' methods

Ex:

```python
x, sr = sf.load('file.wav')
model = Model()
isolated, background = model.forward(x)
sf.write('isolated.wav', isolated, sr)
```