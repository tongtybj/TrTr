from glob import glob
import numpy as np
import os
import shutil
import argparse


def pack_trackingnet_results(tracker_name, model_name):

    results_path = os.path.join(tracker_name, 'TrackingNet', model_name)
    if not os.path.exists(results_path):
        print("can not find {}".format(results_path))
        return

    output_path = os.path.join(results_path, 'converted')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    seqs = glob(os.path.join(results_path, "*.txt"))
    assert len(seqs) == 511


    for seq in seqs:
        seq_name = os.path.basename(seq)
        results = np.loadtxt(seq, dtype=np.float64, delimiter=',')

        np.savetxt(os.path.join(output_path, seq_name), results, delimiter=',', fmt='%.2f')

    # Generate ZIP file
    shutil.make_archive(model_name, 'zip', output_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('view tracking result', add_help=False)
    parser.add_argument('--model_name', default="", type=str, help='name of tracker model')
    parser.add_argument('--tracker_path', default="", type=str, help='path of tracker result')
    args = parser.parse_args()

    pack_trackingnet_results(args.tracker_path, args.model_name)
