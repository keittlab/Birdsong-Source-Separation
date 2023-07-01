# Adjust these settings ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####

BATCH_SIZE = 15
NUM_WORKERS = 1
SEGMENT_LENGTH = 3  # sec # training input size
SAMPLES_PER_EPOCH = 1000  # 80% training 20% validation
PRELOAD_AUDIO = False  # preloads audio into memory before generation
CHECKPOINT_PATH = 'Checkpoints/SUDO48-3sec.ckpt'

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####

TRAINING_PERCENTAGE = .05  # percent of all data to train on if not JIT_GENRATION
JIT_GENRATION = True  # generated audio just in time instead of generating beforehand
# Read data from these paths if the data has been pre generated
GENERATED_READ_DATA_PATHS = ['', '']
# Write to this path when generating data
GENERATED_WRITE_DATA_PATH = ''
# Read data from this path to generate the data
RAW_DATA_PATH = 'Data' 
VIRTUAL_FOREST_IR_PATH = 'Virtual Forest IRs' # path to impulse responses
SR = 22050
