from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import torch.utils.data as data
from ssd import build_ssd
from data import BaseTransform

import glob
import cv2

##############################################################################################
IMG_SUFFIX = '*.jpg'

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--test_src_path', default='data/test_src',
                    type=str, help='Location of VOC root directory')
parser.add_argument('--trained_model', default='weights/VOC.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='test_result/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda to train model')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()
if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
##############################################################################################

def test_net(save_folder, net, cuda, all_src_img, transform, thresh):
    num_images = len(all_src_img)
    for count in range(num_images):
        print('Testing image {}/{}....'.format(count+1, all_src_img[count]))
        img = cv2.imread(all_src_img[count], cv2.IMREAD_COLOR)
        # to rgb, maybe useless !
        # img = img(2, 1, 0)
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))

        if cuda:
            x = x.cuda()

        y = net(x)  # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= thresh:
                score = detections[0, i, j, 0]
                label_name = VOC_CLASSES[i-1]
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                j += 1

                # draw results and save result images
                cv2.rectangle(img, (coords[0], coords[1]), (coords[2], coords[3]),
                              (0, 0, 255), thickness=1)
                strText = label_name + ': ' + str(round(score.item(), 2))
                cv2.putText(img, strText, (coords[0], coords[1]),
                           cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 1)
        file_temp = all_src_img[count].replace('\\', '/')
        file_name = file_temp.split('/')[-1]
        file_name = os.path.join(save_folder, file_name)
        cv2.imwrite(file_name, img)
    print('Finished the testing task !')

def main():
    # load net
    num_classes = 20 + 1    # +1 background
    net = build_ssd('test', 300, num_classes)   # initialize SSD
    net.load_state_dict(torch.load(args.trained_model, map_location='cpu'))
    net.eval()
    print('Finished loading model!')
    # load data
    all_src_img = glob.glob(os.path.join(args.test_src_path, IMG_SUFFIX))
    if len(all_src_img) < 1:
        print("Have not image for testing !")
        assert(0)
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # infer main function
    test_net(args.save_folder, net, args.cuda, all_src_img,
             BaseTransform(net.size, (104, 117, 123)),
             thresh=args.visual_threshold)

if __name__ == '__main__':
    main()
