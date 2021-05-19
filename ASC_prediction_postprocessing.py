import os
import sys
import csv
import glob
import numpy as np
from scipy.signal import medfilt
from scipy.special import softmax


def write_to_file(all_data, target):
    with open(target, mode='w') as ef:
        efw = csv.writer(ef, delimiter=',')
        for data in all_data:
            efw.writerow(data)


def select_files(pred_source, gt_source):
    pred_files = glob.glob(pred_source+'/*.csv')
    pred_files.sort()

    gt_files = glob.glob(gt_source+'/*.csv')
    gt_files.sort()

    return pred_files, gt_files

def softmax_feats(source, filter_lenght):
    print(source)
    data = csv_to_list(source)

    positive_predictions = []
    for d in data:
        positive_predictions.append( [float(d[-2]), float(d[-1])] )
    positive_predictions = np.asarray(positive_predictions)

    positive_predictions[..., 0] = medfilt(positive_predictions[..., 0], filter_lenght)
    positive_predictions[..., 1] = medfilt(positive_predictions[..., 1], filter_lenght)
    positive_predictions = softmax(positive_predictions, axis = -1)

    for idx in range(len(data)):
        row = data[idx]
        del row[-2]
        row[-1] = float(positive_predictions[idx][1])
    return data

def csv_to_list(csv_path):
    as_list = None
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        as_list = list(reader)
    return as_list


if __name__ == '__main__':
    predicitions_dir = '...' #directory with network predictions
    ava_csv_files = '.../AVA/csv/val' # directory with original ava csv files

    temporary_dir = '.../tmp' #Any EMPTY directory
    target_csv_pred = '.../ASCPredictions.csv' # Final prediction file
    target_csv_gt = '.../gt.csv' #utility file to use the official evaluation tool


    del_files = glob.glob(temporary_dir+'/*')
    for f in del_files:
        os.remove(f)

    pred_files, gt_files = select_files(predicitions_dir, ava_csv_files)
    for idx, pf in enumerate(pred_files):
        pred_data = csv_to_list(pf)
        gt_data = csv_to_list(os.path.join(ava_csv_files, os.path.basename(pf)[:-4]+'-activespeaker.csv'))

        print(idx, os.path.basename(pf), len(pred_data), len(gt_data))
        post_processed_data = softmax_feats(pf, 1)

        for idx in range(len(post_processed_data)):
            post_processed_data[idx] = [gt_data[idx][0], gt_data[idx][1],
                                        gt_data[idx][2], gt_data[idx][3],
                                        gt_data[idx][4], gt_data[idx][5],
                                        'SPEAKING_AUDIBLE', gt_data[idx][-1],
                                        '{0:.4f}'.format(post_processed_data[idx][-1])]

        target_csv = os.path.join(temporary_dir, os.path.basename(pf))
        write_to_file(post_processed_data, target_csv)

    processed_gt_files = glob.glob(temporary_dir+'/*.csv')
    processed_gt_files.sort()
    gt_files.sort()
    os.system('cat ' + ' '.join(processed_gt_files) + '> '+ target_csv_pred)
    os.system('cat ' + ' '.join(gt_files) + '> '+ target_csv_gt)
