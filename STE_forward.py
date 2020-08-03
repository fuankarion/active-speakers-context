import os
import csv
import sys
import torch

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from core.dataset import AudioVideoDatasetAuxLossesForwardPhase
from core.optimization import optimize_av_losses
from core.io import set_up_log_and_ws_out
from core.util import configure_backbone_forward_phase, load_train_video_set, load_val_video_set

import core.custom_transforms as ct
import core.config as exp_conf


#Written for simplicity, paralelize/shard as you wish
if __name__ == '__main__':
    clip_lenght = int(sys.argv[1])
    cuda_device_number = str(sys.argv[2])
    image_size = (144, 144) #Dont forget to assign this same size on ./core/custom_transforms

    model_weights = '.../ste_encoder_big/99.pth'
    target_directory = '.../train_forward/'
    io_config = exp_conf.STE_inputs
    opt_config = exp_conf.STE_forward_params
    opt_config['batch_size'] = 1

    # cuda config
    backbone = configure_backbone_forward_phase(opt_config['backbone'], model_weights, clip_lenght)
    has_cuda = torch.cuda.is_available()
    device = torch.device('cuda:'+cuda_device_number if has_cuda else 'cpu')
    backbone = backbone.to(device)

    video_data_transforms = {
        'val': ct.video_val
    }

    video_val_path = os.path.join(io_config['video_dir'], 'train')
    audio_val_path = os.path.join(io_config['audio_dir'], 'train')

    #train_videos = load_train_video_set()
    val_videos =  load_val_video_set()

    for video_key in val_videos:
        print('forward video ', video_key)
        with open(target_directory+video_key+'.csv', mode='w') as vf:
            vf_writer = csv.writer(vf, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            d_val = AudioVideoDatasetAuxLossesForwardPhase(video_key, audio_val_path, video_val_path,
                                            io_config['csv_train_full'], clip_lenght,
                                            image_size, video_data_transforms['val'],
                                            do_video_augment=False)

            dl_val = DataLoader(d_val, batch_size=opt_config['batch_size'],
                                shuffle=False, num_workers=opt_config['threads'])

            for idx, dl in enumerate(dl_val):
                print(' \t Forward iter ', idx, '/', len(dl_val), end='\r')
                audio_data, video_data, video_id, ts, entity_id, gt = dl
                video_data = video_data.to(device)
                audio_data = audio_data.to(device)

                with torch.set_grad_enabled(False):
                    preds, _, _, feats = backbone(audio_data, video_data)
                    feats = feats.detach().cpu().numpy()[0]
                    vf_writer.writerow([video_id[0], ts[0], entity_id[0], float(gt[0]), float(preds[0][0]), float(preds[0][1]), list(feats)])
