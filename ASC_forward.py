#TODO two files for this one for train one for val
import os
import csv
import sys
import torch

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from core.dataset import ASCFeaturesDatasetForwardPhase
from core.optimization import optimize_av_losses
from core.models import ASC_Net
from core.io import set_up_log_and_ws_out
from core.util import configure_backbone_forward_phase, load_train_video_set, load_val_video_set

import core.custom_transforms as ct
import core.config as exp_conf


def select_files(pred_source, gt_source):
    pred_files = glob.glob(pred_source+'/*.csv')
    pred_files.sort()

    gt_files = glob.glob(gt_source+'/*.csv')
    gt_files.sort()

    return pred_files, gt_files


#Written for simplicity, paralelize/shard as you wish
if __name__ == '__main__':
    clips = int(sys.argv[1])
    time_stride = int(sys.argv[2])
    speakers = int(sys.argv[3])
    cuda_device_number = str(sys.argv[4])

    model_weights = '...'
    target_directory = '...'
    io_config = exp_conf.ASC_inputs_forward
    opt_config = exp_conf.ASC_forward_params

    # cuda config
    has_cuda = torch.cuda.is_available()
    device = torch.device('cuda:'+cuda_device_number if has_cuda else 'cpu')

    backbone = ASC_Net(clip_number=clips, candidate_speakers=speakers )
    backbone.load_state_dict(torch.load(model_weights, map_location='cpu'))
    backbone.eval()
    backbone = backbone.to(device)


    val_videos =  load_val_video_set()

    for video_key in val_videos:
        print('forward video ', video_key)
        with open(os.path.join(target_directory, video_key+'.csv'), mode='w') as vf:
            vf_writer = csv.writer(vf, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            features_file = os.path.join(io_config['features_val_full'], video_key+'.csv')
            d_val = ASCFeaturesDatasetForwardPhase(features_file, clips, time_stride, speakers)

            dl_val = DataLoader(d_val, batch_size=opt_config['batch_size'],
                                shuffle=False, num_workers=opt_config['threads'])

            for idx, dl in enumerate(dl_val):
                print(' \t Forward iter ', idx, '/', len(dl_val), end='\r')
                features, video_id, ts, entity_id = dl
                features = features.to(device)

                with torch.set_grad_enabled(False):
                    pred = backbone(features)
                    pred = pred.detach().cpu().numpy()
                    vf_writer.writerow([entity_id[0], ts[0], str(pred[0][0]), str(pred[0][1])])
