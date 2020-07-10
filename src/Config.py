# Default configuration of ROM

# Data set we are emulating
geo_data = 'tmax' # 'tmax', 'prcp'

# Type of compression
compression = 'pod' # 'pod', 'cae'

# Training durations
num_epochs_time = 2000
num_epochs_space = 100

# Forecast window
window_len = 7

# Train or test for both space and time
train_mode = False

# Deployment mode space
space_train_mode = False

# Deployment mode space
time_train_mode = False

# Batchsize
batchsize_space = 1 # CAE
batchsize_time = 64 # LSTM

# For POD
num_modes = 5
# FOR CAE
num_latent_space = 8

# Field visualization
field_viz = False

# Horovod
use_hvd = False


if __name__ == '__main__':
    pass
