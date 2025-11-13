import os.path as osp
import os
import numpy as np


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


seq_root = "../datasets4MOTIP/vas-mot/train/"
#seq_root = "../datasets4MOTIP/vas-mot/val/"

label_root = "../datasets4MOTIP/vas-mot/labels_with_ids/train/"
#label_root = "../datasets4MOTIP/vas-mot/labels_with_ids/val/"

mkdirs(label_root)
seqs = [s for s in os.listdir(seq_root)]

tid_curr = 0
tid_last = -1
for seq in seqs:
    seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
    print(seq_info)
    seq_width = int(seq_info[seq_info.find('imwidth = ') + 10:seq_info.find('\nimheight')])
    seq_height = int(seq_info[seq_info.find('imheight = ') + 11:seq_info.find('\nimext')])

    gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')
    gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')

    seq_label_root = osp.join(label_root, seq)
    mkdirs(seq_label_root)

    # if gt.shape[1] > 9:
    #     gt = gt[:, [0, 1, 2, 3, 4, 5, 7, 6, 8]]  # 交换第6列(label)和第7列(_)的位置

    # #for fid, tid, x, y, w, h, label, _, _, _ in gt:
    for fid, tid, x, y, w, h, label, _, _, _ in gt:
        fid = int(fid)
        tid = int(tid)
        if not tid == tid_last:
            tid_curr += 1
            tid_last = tid

        label_fpath = osp.join(seq_label_root, "format_gt.txt")
        label_str = '{:d} {:d} {:d} {:d} {:d} {:d} {:f}\n'.format(
            fid, tid_curr, int(x), int(y), int(w), int(h), int(label))  # in int, without normalization
        print (label_str)

        with open(label_fpath, 'a') as f:
            f.write(label_str)