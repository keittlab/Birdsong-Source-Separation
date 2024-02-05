from globalVars import BATCH_SIZE, SR, SEGMENT_LENGTH
from asteroid.models import DPRNNTasNet
from asteroid.models import ConvTasNet
from asteroid.models import SuDORMRFNet
import torch
import numpy as np
from math import ceil


# model class for easy use
class Model:
    def __init__(self, checkpointPath='Checkpoints/SUDO72-3sec.ckpt', modelType='SuDORMRFNet'):
        self.modelType = modelType
        if modelType == 'DPRNNTasNet':
            self.model = DPRNNTasNet(n_src=2)
        elif modelType == 'ConvTasNet':
            self.model = ConvTasNet(n_src=2)
        elif modelType == 'SuDORMRFNet':
            self.model = SuDORMRFNet(n_src=2)
        else:
            raise Exception(f'Invalid model type: \'{modelType}\'')

        # send model to GPU if available
        if torch.backends.mps.is_available():
            print('Using Mac GPU')
            self.model.to(torch.device("mps"))
            
        elif torch.cuda.is_available():
            print('Using CUDA')
            self.model.to(torch.device("cuda"))

        # load checkpoint
        checkpoint = torch.load(checkpointPath, map_location='cpu')
        state_dict = {}
        for key in checkpoint['state_dict']:
            state_dict[key.replace('model.', '')] = checkpoint['state_dict'][key]
        self.model.load_state_dict(state_dict)

        self.model.eval()
        torch.set_grad_enabled(False)


    # takes in single channel audio and returns 2 sources
    def forward(self, input):
        # space segments to ignore 1 sec at start and end
        segmentSpacing = (SEGMENT_LENGTH - 2) * SR
        # find num batches needed
        numBatches = ceil(len(input) / segmentSpacing)
        # creates a long enough array to hold all batches
        newLen = numBatches * segmentSpacing + 2 * SR  
        # add buffer to beginning and end
        buffSize = ceil((newLen - len(input)) / 2)  
        # add buffer of silence to beginning and end
        padded = np.pad(input, buffSize, 'constant', constant_values=(0, 0))
        
        # splits data into batches
        combined = np.empty((numBatches, SEGMENT_LENGTH * SR), dtype=np.float32)
        for i in range(numBatches):
            start = i * segmentSpacing
            end = start + SEGMENT_LENGTH * SR
            combined[i] = padded[start:end]
            
        # run model on batches
        estimated_sources = np.empty((numBatches, 2, SEGMENT_LENGTH * SR), dtype=np.float32)
        for i in range(ceil(len(combined) / BATCH_SIZE)):
            start = i * BATCH_SIZE
            end = (i + 1) * BATCH_SIZE
            end = min(end, len(combined))  # don't go out of bounds
            modelInput = torch.from_numpy(combined[start:end])
            
            # use mac GPU if available
            if torch.backends.mps.is_available():
                modelInput = modelInput.to('mps')
                
            # use CUDA if available
            elif torch.cuda.is_available():
                modelInput = modelInput.to('cuda')
            modelOutput = self.model.forward(modelInput)
            del modelInput
            estimated_sources[start:end] = modelOutput.to('cpu').detach().numpy()
            
        # calculate fade mask for overlapping segments
        fadeMask = np.zeros(SEGMENT_LENGTH * SR)
        for i in range(int(.5 * SR)):
            fadeMask[i + int(.75 * SR)] = i / (.5 * SR)
            fadeMask[i + int((SEGMENT_LENGTH - 1.25) * SR)] = 1 - (i / (.5 * SR))
        fadeMask[int(1.25 * SR):-int(1.25 * SR)] = 1
        
        # combine batches into output using fade mask to overlap segments
        output = np.zeros((2, newLen), dtype=np.float32)
        for i in range(numBatches):
            start = i * segmentSpacing
            end = start + SEGMENT_LENGTH * SR
            if self.modelType == 'SuDORMRFNet':
                output[0, start:end] += estimated_sources[i, 0] * fadeMask  # don't switch background and isolated
                output[1, start:end] += estimated_sources[i, 1] * fadeMask
            else:
                output[1, start:end] += estimated_sources[i, 0] * fadeMask  # switch background and isolated
                output[0, start:end] += estimated_sources[i, 1] * fadeMask
        
        # remove buffer and return
        return output[:, buffSize: -buffSize]