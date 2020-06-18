# Active Speakers in Context
This repo contains the official code and models for the "Active Speakers in Context" CVPR 2020 [paper](https://arxiv.org/pdf/2005.09812.pdf).


## Before Training
This code works over  face crops and their corresponding audio track, before you start training you need to preprocess the videos in the AVA dataset:

1. Extract the audio tracks from every video in the dataset. Go to ./data/extract_audio_tracks.py in  __main__ adapt the `ava_video_dir` (directory with the original ava videos) and `target_audios` (empty directory where the audio tracks will be stores) to your local file system. The code relies on 16k wav files and will fail with other formats and bit rates.
2. Slice the audio tracks by timestamp. Go to ./data/slice_audio_tracks.py in  __main__ adapt the `ava_audio_dir` (the audio tracks you extracted on step 1), `output_dir` (empty directory where you will store the sliced audio files) and  `csv` (....) to your local file system.
3. Extract the face crops by timestamp. Go to ./data/extract_audio_tracks.py in  __main__ adapt the `ava_video_dir` (directory with the original ava videos), `csv_file` (....) and  `output_dir` (empty directory where you will store the face crops) to your local file system.

The full audio tracks obtained on 1 step 1. will not be used anyore.

## Training
Training the Active Speaker Context is divided in two major stages: the optimization of the Short-Term Encoder and the optimization of the Context Ensemble Network. The second step includes the pair-wise refinement and the temporal refinement, and relies on a full forward pass of the Short-Term Encoder on the training and validation sets.

### Training the Short-Term Encoder
Got to ./core/config.py  and modify the ` STE_inputs`  dictionary so that the keys audio_dir, video_dir and models_out point to the directories with the extracted audio clips, face crops and an empty directory where the STE models will be saved.

Execute the script python `STE_train.py clip_lenght cuda_device_number`, we used clip_lenght=11 on the paper, it can be set to any uneven value greater than 0 but performance will vary.

### Forward Short Term Encoder
Training on the Active Speaker context relies on extracting high-level features from the STE over the full active speaker set. Execute the script `python STE_forward.py clip_lenght cuda_device_number`, **use the same clip_lenght as the training**.

The STE_forward.py relies on the same input files as the STE_train.py script, no extra adjustment should be required. To switch between training and val sets go to lines 44 and 45 to obtain a list of training and val videos, you will need both subsets for the next step.

### Training the Active Speaker Context
Got to ./core/config.py  and modify the ` ASC_inputs`  dictionary so that the keys features_train_full, features_val_full and  point to the directories with the extracted  short term encoder features in training and validations sets, and an empty directory where the  models will be stored

Execute the script python `ASC_train.py  time_lenght time_stride speakers cuda_device_number`, in the paper we used time_lenght=11 time_stride=4  and speakers=3, these params can be se to any  value greater than 0 but performance will vary.

### Forward the AS Model
As final step use the script ‘ASC_forward.py’ go to lines 22 and 23 and select the model weights and the directory where you will forward the final predictions.


# Using the Docker image
Coming soon...


# Official models
If you just want to use the active speakers in context models in forward phase download our models here

Coming soon...
