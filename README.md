# Active Speakers in Context
This repo contains the official code and models for the "Active Speakers in Context" CVPR 2020 [paper](https://arxiv.org/pdf/2005.09812.pdf).


## Before Training
This code works over  face crops and their corresponding audio track, before you start training you need to preprocess the videos in the AVA dataset. We have 3 utility files that contain the basic data to start this process, download them using `./scripts/dowloads.sh`.

1. Extract the audio tracks from every video in the dataset. Go to ./data/extract_audio_tracks.py in  __main__ adapt the `ava_video_dir` (directory with the original ava videos) and `target_audios` (empty directory where the audio tracks will be stores) to your local file system. **The code relies on 16k wav files and will fail with other formats and bit rates**.
2. Slice the audio tracks by timestamp. Go to ./data/slice_audio_tracks.py in  __main__ adapt the `ava_audio_dir` (the audio tracks you extracted on step 1), `output_dir` (empty directory where you will store the sliced audio files) and  `csv` (the utility file you download previously, use the set accordingly) to your local file system.
3. Extract the face crops by timestamp. Go to ./data/extract_audio_tracks.py in  __main__ adapt the `ava_video_dir` (directory with the original ava videos), `csv_file` (the utility file you download previously, use the set accordingly) and  `output_dir` (empty directory where you will store the face crops) to your local file system.

The full audio tracks obtained on 1 step 1. will not be used anymore.

## Training
Training the ASC is divided in two major stages: the optimization of the Short-Term Encoder (similar to google [baseline](https://arxiv.org/abs/1901.01342)) and the optimization of the Context Ensemble Network. The second step includes the pair-wise refinement and the temporal refinement, and relies on a full forward pass of the Short-Term Encoder on the training and validation sets.

### Training the Short-Term Encoder
Got to ./core/config.py  and modify the STE_inputs dictionary so that the keys audio_dir, video_dir and models_out points to to the audio clips, face crops (those extracted on ‘Before Training’) and an empty directory where the STE models will be saved.

Execute the script `STE_train.py clip_lenght cuda_device_number`, we used clip_lenght=11 on the paper, but it can be set to any uneven value greater than 0 (performance will vary).

### Forward Short Term Encoder
The Active Speaker context relies on the features extracted from the STE, execute the script python `STE_forward.py clip_lenght cuda_device_number`, use the same clip_lenght as the training. Check lines 44 and 45 to obtain a list of training and val videos, you will need both subsets for the next step

### Training the ASC Module
Once all the STE features have been calculated, go to ./core/config.py and change the dictionary ‘ASC_inputs’ and modify the value of keys, features_train_full, features_val_full, and  models_out point to point to the local directories where the features train and val have been stored and an empty directory where the models will 'be stored.  execure ‘./ASC_train.py clip_lenght skip_frames speakers cuda_device_number’ clip_lenght must be the same size used to train the STE, skip_frames determines the amount of frames in between sampled clips, we used 4 for the results presented in the paper, speakers is the number of candidates speakers in the contex.

### Forward ASC
use ./ASC_forward.py to forward the models produced by the last step


### Pretrained Models
[Short Term Encoder](https://filedn.com/l0kNCNuXuEq70c3iUHsXxJ7/active-speakers-context/STE.pth) 
[Active Speaker Context](https://filedn.com/l0kNCNuXuEq70c3iUHsXxJ7/active-speakers-context/ASC.pth)
