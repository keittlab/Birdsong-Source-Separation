# Data Cleaner Settings ##### ##### ##### ##### ##### ##### ##### #####

# Bird to extract from raw recordings (case insensitive)
# see speciesList.csv for options
BIRD_OF_INTEREST = 'Golden-Cheeked Warbler'

# Min frequency for bird of interest
# Used to remove low frequency noise
BIRD_OF_INTEREST_MIN_FREQ = 2000  # Hz

# Probability distribution each folder within Data of being selected during sample generation
DISTRIBUTION = {
    'Positive Samples': 100,
    'Background Samples/Volume Based': 40,  # loudest 1 minute of each raw recording
    'Background Samples/Label Based': 60,  # highest classificaiton probability for each category
}

# Number of times to perform noise reduction to clean birdcalls
NOISE_REDUCTION_ITERATIONS = 5  # 5 is good balance between noise and birdcall quality

# Minimum threshold for classification
THRESHOLD = 0.2

# Max input length for PANNS classifier
# Use to reduce memory usage
PANNS_BATCH_SIZE_IN_MINUTES = 30  # minutes

# Training Settings ##### ##### ##### ##### ##### ##### ##### #####

# If training, resume training from this checkpoint
# If evaluating, evaluate this checkpoint
CHECKPOINT_PATH = 'Checkpoints/SUDO48-3sec.ckpt'

# Number of samples to generate per epoch
SAMPLES_PER_EPOCH = 1000

# Number of samples to train on per batch
BATCH_SIZE = 15

# Number of workers generating samples
NUM_WORKERS = 1

# model input size
SEGMENT_LENGTH = 3  # sec

# preloads audio into memory before generation
# this is faster but requires more memory
PRELOAD_AUDIO = False

# Advanced Settings ##### ##### ##### ##### ##### ##### ##### #####

# Audio Sample Rate
# all files will be resampled to this sample rate and converted to mono audio
SR = 22050

# generated audio just in time instead of generating beforehand
# this is faster because it doesn't require disk access
JIT_GENRATION = True

# Read data from this path to generate the data
RAW_DATA_PATH = 'Data'

# Path to impulse responses
VIRTUAL_FOREST_IR_PATH = 'Virtual Forest IRs'

# percent of all data to train per epoch on if not JIT_GENRATION
TRAINING_PERCENTAGE = 1.0

# Read data from these paths if the data has been pre generated
GENERATED_READ_DATA_PATHS = ['', '']

# Write to this path when pre-generating data
GENERATED_WRITE_DATA_PATH = ''

# Fixed Species Name ##### ##### ##### ##### ##### ##### ##### #####
if 'haveFixedSpeciesName' not in locals():
    from birdnetlib.species import SpeciesList

    speciesListLoader = SpeciesList()
    speciesListLoader.load_labels()
    speciesList = [species.split('_')[1] for species in speciesListLoader.labels]
    speciesMap = {species.lower(): species for species in speciesList}

    BIRD_OF_INTEREST = BIRD_OF_INTEREST.lower().strip()

    if BIRD_OF_INTEREST not in speciesMap:
        print(speciesList)
        raise Exception('Bird of interest not found in above species list')

    BIRD_OF_INTEREST = speciesMap[BIRD_OF_INTEREST]
    haveFixedSpeciesName = True