import os
import csv
import glob

import numpy as np
from scipy.special import softmax
from scipy.signal import medfilt

from core.io import csv_to_list

def write_to_file(all_data, target):
    with open(target, mode='w') as ef:
        efw = csv.writer(ef, delimiter=',')
        for data in all_data:
            efw.writerow(data)

def prediction_postprocessing(data, filter_lenght):
    positive_predictions = []
    for d in data:
        positive_predictions.append( [float(d[-3]), float(d[-2])] )
    positive_predictions = np.asarray(positive_predictions)

    positive_predictions[..., 0] = medfilt(positive_predictions[..., 0], filter_lenght)
    positive_predictions[..., 1] = medfilt(positive_predictions[..., 1], filter_lenght)
    positive_predictions = softmax(positive_predictions, axis = -1)

    for idx in range(len(data)):
        row = data[idx]
        del row[-2]
        row[-1] = float(positive_predictions[idx][1])
    return data

def select_files(pred_source, gt_source):
    pred_files = glob.glob(pred_source+'/*.csv')
    pred_files.sort()

    gt_files = glob.glob(gt_source+'/*.csv')
    gt_files.sort()

    return pred_files, gt_files


if __name__ == '__main__':
    forward_dir = '.../Forwards/ActiveSpeakers/publish' #Directory where you store the network predcitons
    ava_ground_truth_dir = '.../AVA/csv/val' #AVA original ground truth files
    temporary_dir = '.../temp/activeSpeakers' #Just an empty temporary dir

    # The script will generate these two, use them for the official AVA evaluation
    dataset_predictions_csv = '.../Forwards/ActiveSpeakers/publish/final/STE.csv'  #file with final predictions
    dataset_gt_csv = '...Forwards/ActiveSpeakers/publish/final/gt.csv' # Utility file to use the official evaluation tool

    #cleanup temp dir
    del_files = glob.glob(temporary_dir+'/*')
    for f in del_files:
        os.remove(f)

    pred_files, gt_files = select_files(forward_dir, ava_ground_truth_dir)

    for idx, (pf, gtf) in enumerate(zip(pred_files, gt_files)):
        prediction_data = csv_to_list(pf)
        gt_data = csv_to_list(gtf)

        print('Match', os.path.basename(pf), len(prediction_data), len(gt_data))
        if len(prediction_data)!= len(gt_data):
            raise Exception('Groundtruth and prediction dont match in lenght')

        post_processed_predictions = prediction_postprocessing(prediction_data, 1)

        #reformat into ava required style
        for idx in range(len(post_processed_predictions)):
            post_processed_predictions[idx] = [gt_data[idx][0], gt_data[idx][1],
                                        gt_data[idx][2], gt_data[idx][3],
                                        gt_data[idx][4], gt_data[idx][5],
                                        'SPEAKING_AUDIBLE', gt_data[idx][-1],
                                        '{0:.4f}'.format(post_processed_predictions[idx][-1])]

        target_csv = os.path.join(temporary_dir, os.path.basename(pf))
        write_to_file(post_processed_predictions, target_csv)

    processed_gt_files = glob.glob(temporary_dir+'/*.csv')
    processed_gt_files.sort()
    gt_files.sort()
    os.system('cat ' + ' '.join(processed_gt_files) + '> '+ dataset_predictions_csv)
    os.system('cat ' + ' '.join(gt_files) + '> '+ dataset_gt_csv)
