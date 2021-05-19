# Active Speakers in Context
This repo contains the official code and models for the "Active Speakers in Context" CVPR 2020 [paper](https://arxiv.org/pdf/2005.09812.pdf).


## Before Training
The code relies on multiple external libraries go to `./scripts/dev_env.sh`.an recreate the suggested envirroment.

This code works over face crops and their corresponding audio track, before you start training you need to preprocess the videos in the AVA dataset. We have 3 utility files that contain the basic data to support this process, download them using `./scripts/dowloads.sh`.

1. Extract the audio tracks from every video in the dataset. Go to ./data/extract_audio_tracks.py in  __main__ adapt the `ava_video_dir` (directory with the original ava videos) and `target_audios` (empty directory where the audio tracks will be stored) to your local file system. **The code relies on 16k .wav files and will fail with other formats and bit rates**.
2. Slice the audio tracks by timestamp. Go to ./data/slice_audio_tracks.py in  __main__ adapt the `ava_audio_dir` (the directory with the audio tracks you extracted on step 1), `output_dir` (empty directory where you will store the sliced audio files) and  `csv` (the utility file you download previously, use the set accordingly) to your local file system.
3. Extract the face crops by timestamp. Go to ./data/extract_face_crops_time.py in  __main__ adapt the `ava_video_dir` (directory with the original ava videos), `csv_file` (the utility file you download previously, use the train/val/test set accordingly) and  `output_dir` (empty directory where you will store the face crops) to your local file system. This process will result in about 124GB extra data.

The full audio tracks obtained on step 1. will not be used anymore.

## Training
Training the ASC is divided in two major stages: the optimization of the Short-Term Encoder (similar to google [baseline](https://arxiv.org/abs/1901.01342)) and the optimization of the Context Ensemble Network. The second step includes the pair-wise refinement and the temporal refinement, and relies on a full forward pass of the Short-Term Encoder on the training and validation sets.

### Training the Short-Term Encoder
Got to ./core/config.py  and modify the `STE_inputs` dictionary so that the keys `audio_dir`, `video_dir` and `models_out` point to the audio clips, face crops (those extracted on ‘Before Training’) and an empty directory where the STE models will be saved.

Execute the script `STE_train.py clip_lenght cuda_device_number`, we used clip_lenght=11 on the paper, but it can be set to any uneven value greater than 0 (performance will vary!).

### Forward Short Term Encoder
The Active Speaker Context relies on the features extracted from the STE for its optimization, execute the script python `STE_forward.py clip_lenght cuda_device_number`, use the same clip_lenght as the training. Check lines 44 and 45 to switch between a list of training and val videos, **you will need both subsets** for the next step.

If you want to evaluate on the AVA Active Speaker Datasets, use ./STE_postprocessing.py, check lines 44 to 50 and adjust the files to your local file system.

### Training the ASC Module
Once all the STE features have been calculated, go to `./core/config.py` and change the dictionary `ASC_inputs` modify the value of keys, `features_train_full`, `features_val_full`, and `models_out` so that they point to the local directories where the features extracted with the STE in the train and val set have been stored, and an empty directory where the  ASC models will 'be stored.  Execute `./ASC_train.py clip_lenght skip_frames speakers cuda_device_number` clip_lenght must be the same clip size used to train the STE, skip_frames determines the amount of frames in between sampled clips, we used 4 for the results presented in the paper, speakers is the number of candidates speakers in the contex.

### Forward ASC
use `./ASC_forward.py clips time_stride speakers cuda_device_number` to forward the models produced by the last step. Use the same clip and stride configurations. You will get one csv file for every video, for evaluation purposes use the script ASC_predcition_postprocessing.py to generate a single CSV file which is compatible with the evaluation tool, check lines 54 to 59 and adapt the paths to your local configuration.

If you want to evaluate on the AVA Active Speaker Datasets, use ./ASC_prediction_postprocessing.py, check lines 54 to 59 and adjust the files to your local file system.

### Pre-Trained Models
[Short Term Encoder](https://filedn.com/l0kNCNuXuEq70c3iUHsXxJ7/active-speakers-context/STE.pth) 

[Active Speaker Context](https://filedn.com/l0kNCNuXuEq70c3iUHsXxJ7/active-speakers-context/ASC.pth)


### Prediction Postprocessing and Evaluation
The prediction format follows the very same format of the AVA-Active speaker dataset, but contains an extra value for the active speaker class in the final column. The script `./STE_postprocessing.py` handles this step. Check lines 44, 45 and 46 and set the directory where you saved the output of the forward pass (44), the directory with the original ava csv (45) and and empty temporary directory (46). Additionally set on lines 48 and 49 the outputs of the script, one of them is the final prediction formated to use the official evaluation tool and the other one is a utility file to use along the same tool.
Notice you can do some temporal smoothing on the function 'softmax_feats', is a simple median filter and you can choose the window size on lines 35 and 36. 
