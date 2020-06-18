# Active Speakers in Context
This repo contains the official code and models for the "Active Speakers in Context" CVPR 2020 [paper](https://arxiv.org/pdf/2005.09812.pdf).


## Before Training
This code works over  face crops and their corresponding audio track, before you start training you need to preprocess the videos in the AVA dataset:

1. Extract the audio tracks from every video in the dataset. Go to ./data/extract_audio_tracks.py in  __main__ adapt the `ava_video_dir` (directory with the original ava videos) and `target_audios` (empty directory where the audio tracks will be stores) to your local file system. The code relies on 16k wav files and will fail with other formats and bit rates.
2. Slice the audio tracks by timestamp. Go to ./data/slice_audio_tracks.py in  __main__ adapt the `ava_audio_dir` (the audio tracks you extracted on step 1), `output_dir` (empty directory where you will store the sliced audio files) and  `csv` (....) to your local file system.
3. Extract the face crops by timestamp. Go to ./data/extract_audio_tracks.py in  __main__ adapt the `ava_video_dir` (directory with the original ava videos), `csv_file` (....) and  `output_dir` (empty directory where you will store the face crops) to your local file system.

The full audio tracks obtained on 1 step 1. will not be used anyore.

## Training
Trainign the ASC is divided in two major stages: the optimization of the Short-Term Encoder and the optimization of the Context Ensenmble Network. The second step includes the pair-wise refinement and the temporal refinement, and relies on a full forward pass of the Short-Term Encoder on the training and validation sets.

### Training the Short-Term Encoder
Got to ./core/congi.py  and modify the STE_inputs dictionary so that the keys audio_dir, video_dir and models_out points to to the audio clips, face crops and an empty directory where the models will be saved.

Execute the script python `STE_train.py clip_lenght cuda_device_number`, we used clip_lenght=11 on the paper, but it can be set to any uneven value greater than 0.

### Foward Short Tem Encoder
The Active Speaker context relies on the features extracted from the STE, execute the script python `STE_forward.py clip_lenght cuda_device_number`, use the same clip_lenght as the training. Check lines 44 and 45 to obtaina list of training and val videos, you will need both subsets for the next step


# Using the Docker image
The docker image is the recommended way to run this code, everything is packed in the image and this most recent version of this code should run out of the box.

Donwload the docker iage from docker hub at ....., download this repo somehwre on the image file system.

train the Short-Term Encoder using the `STE_train.py` script you will find the models on the (.....) directory
