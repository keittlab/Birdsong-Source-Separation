from torch import optim
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
# We train the same model architecture that we used for inference above.
from asteroid.models import ConvTasNet as SelectedModel

# In this example we use Permutation Invariant Training (PIT) and the SI-SDR loss.
from asteroid.losses import pairwise_neg_sisdr, PITLossWrapper
# MiniLibriMix is a tiny version of LibriMix (https://github.com/JorisCos/LibriMix),
# which is a free speech separation dataset.
from asteroid.data import BirdsongDataset

# Asteroid's System is a convenience wrapper for PyTorch-Lightning.
from asteroid.engine import System
from globalVars import BATCH_SIZE, NUM_WORKERS, SEGMENT_LENGTH, SAMPLES_PER_EPOCH, PRELOAD_AUDIO, JIT_GENRATION, CHECKPOINT_PATH, TRAINING_PERCENTAGE, GENERATED_READ_DATA_PATHS, GENERATED_WRITE_DATA_PATH, RAW_DATA_PATH, VIRTUAL_FOREST_IR_PATH, SR, DISTRIBUTION_PATH
from dataGenerator import initGenerator

import os
import torch


if __name__ == "__main__":
    print('ConvTasNet')
    print('BATCH_SIZE: ', BATCH_SIZE)
    print('NUM_WORKERS: ', NUM_WORKERS)
    print('SEGMENT_LENGTH: ', SEGMENT_LENGTH)
    print('SAMPLES_PER_EPOCH: ', SAMPLES_PER_EPOCH)
    print('PRELOAD_AUDIO: ', PRELOAD_AUDIO)
    print('CHECKPOINT_PATH: ', CHECKPOINT_PATH)
    print('TRAINING_PERCENTAGE: ', TRAINING_PERCENTAGE)
    print('JIT_GENRATION: ', JIT_GENRATION)
    print('GENERATED_READ_DATA_PATHS: ', GENERATED_READ_DATA_PATHS)
    print('GENERATED_WRITE_DATA_PATH: ', GENERATED_WRITE_DATA_PATH)
    print('RAW_DATA_PATH: ', RAW_DATA_PATH)
    print('VIRTUAL_FOREST_IR_PATH: ', VIRTUAL_FOREST_IR_PATH)
    print('SR: ', SR)
    print('DISTRIBUTION_PATH: ', DISTRIBUTION_PATH)
    generatorVars = initGenerator()
    torch.cuda.empty_cache()

    train_loader = DataLoader(BirdsongDataset(False, **generatorVars),
                              batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(BirdsongDataset(True, **generatorVars), batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS, drop_last=True)

    # Tell DPRNN that we want to separate to 2 sources.
    model = SelectedModel(n_src=2)

    # Use Mac GPU if available (CUDA also works)
    if torch.backends.mps.is_available():
        print('Using Mac GPU')
        model.to(torch.device("mps"))
        
    # Use CUDA if available
    if torch.cuda.is_available():
        print('Using CUDA')
        model.to(torch.device("cuda"))

    # load checkpoint path
    if CHECKPOINT_PATH != None:
        checkpoint = torch.load(CHECKPOINT_PATH)
        state_dict = {}
        for key in checkpoint['state_dict']:
            state_dict[key.replace('model.', '')] = checkpoint['state_dict'][key]
        model.load_state_dict(state_dict)

    # PITLossWrapper works with any loss function.
    loss = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    system = System(model, optimizer, loss, train_loader, val_loader)

    # Train for 1 epoch using a single GPU. If you're running this on Google Colab,
    # be sure to select a GPU runtime (Runtime Change runtime type  Hardware accelarator).
    trainer = Trainer(max_epochs=200)
    trainer.fit(system)
