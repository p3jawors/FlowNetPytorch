import argparse
from path import Path
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import models
from tqdm import tqdm

import torchvision.transforms as transforms
import flow_transforms
from imageio import imread, imwrite
from imageio.core.util import Array as imageio_array
import numpy as np
from util import flow2rgb
from PIL import Image

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))


parser = argparse.ArgumentParser(description='PyTorch FlowNet inference on a folder of img pairs',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data', metavar='DIR',
                    help='path to images folder, image names must match \'[name]0.[ext]\' and \'[name]1.[ext]\'')
parser.add_argument('pretrained', metavar='PTH', help='path to pre-trained model')
parser.add_argument('--output', '-o', metavar='DIR', default=None,
                    help='path to output folder. If not set, will be created in data folder')
parser.add_argument('--output-value', '-v', choices=['raw', 'vis', 'both'], default='both',
                    help='which value to output, between raw input (as a npy file) and color vizualisation (as an image file).'
                    ' If not set, will output both')
parser.add_argument('--div-flow', default=20, type=float,
                    help='value by which flow will be divided. overwritten if stored in pretrained file')
parser.add_argument("--img-exts", metavar='EXT', default=['png', 'jpg', 'bmp', 'ppm'], nargs='*', type=str,
                    help="images extensions to glob")
parser.add_argument('--max_flow', default=None, type=float,
                    help='max flow value. Flow map color is saturated above this value. If not set, will use flow map\'s max value')
parser.add_argument('--upsampling', '-u', choices=['nearest', 'bilinear'], default=None, help='if not set, will output FlowNet raw input,'
                    'which is 4 times downsampled. If set, will output full resolution flow map, with selected upsampling')
parser.add_argument('--bidirectional', action='store_true', help='if set, will output invert flow (from 1 to 0) along with regular flow')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print('DEVICE IS: ', device)

@torch.no_grad()
def main():
    global args, save_path
    args = parser.parse_args()

    if args.output_value == 'both':
        output_string = "raw output and RGB visualization"
    elif args.output_value == 'raw':
        output_string = "raw output"
    elif args.output_value == 'vis':
        output_string = "RGB visualization"
    print("=> will save " + output_string)
    data_dir = Path(args.data)
    print("=> fetching img pairs in '{}'".format(args.data))
    if args.output is None:
        save_path = data_dir/'flow'
    else:
        save_path = Path(args.output)
    print('=> will save everything to {}'.format(save_path))
    save_path.makedirs_p()
    # Data loading code
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
        transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
    ])

    # print('looking in ', args.data)
    files = os.listdir(args.data)
    # print('all: ', files[:5])
    files[:] = [name for name in files if any(sub in name for sub in args.img_exts)]
    # print('imgs only: ', files[:5])
    files = sorted(files)
    # print('sorted!: ', files[:5])
    img_pairs = []
    for ii in range(0, len(files)-1):
        img1_file = Path('%s/%s' % (args.data, files[ii]))
        img2_file = Path('%s/%s' % (args.data, files[ii+1]))
        img_pairs.append([img1_file, img2_file])
    # print('Have dis many img pairs: ', len(img_pairs))

    # img_pairs = []
    # for ext in args.img_exts:
    #     # test_files = data_dir.files('*1.{}'.format(ext))
    #     test_files = data_dir.files('*0.{}'.format(ext))
    #     for file in test_files:
    #         # img_pair = file.parent / (file.stem[:-1] + '2.{}'.format(ext))
    #         img_pair = file.parent / (file.stem[:-1] + '1.{}'.format(ext))
    #         if img_pair.isfile():
    #             img_pairs.append([file, img_pair])
    #
    # print('{} samples found'.format(len(img_pairs)))

    # create model
    network_data = torch.load(args.pretrained, map_location=torch.device('cuda'))
    print("=> using pre-trained model '{}'".format(network_data['arch']))
    model = models.__dict__[network_data['arch']](network_data).to(device)
    model.eval()
    cudnn.benchmark = True

    if 'div_flow' in network_data.keys():
        args.div_flow = network_data['div_flow']

    nn = 0
    for (img1_file, img2_file) in tqdm(img_pairs):
        # print('img1: ', img1_file)
        # print('img2: ', img2_file)
        # print('img %i/%i' % (nn, len(img_pairs)))
        nn += 1

        #======================= NOT NEEDED AFTER ALL
        # reshape our images because of how network divides and scales data
        # https://github.com/ClementPinard/FlowNetPytorch/issues/11
        # print('OG SHAPE: ', imread(img1_file).shape)
        # print(img1_file)
        # img1 = Image.fromarray(imread(img1_file)).resize((832, 256, 3))
        # img2 = Image.fromarray(imread(img2_file)).resize((832, 256, 3))
        #
        # #NOTE may need to normalize
        # # https://github.com/ClementPinard/FlowNetPytorch/issues/5
        #
        # # convert back to the original imageio array type
        # img1 = np.array(img1)
        # img1 = imageio_array(img1)
        # img2 = np.array(img2)
        # img2 = imageio_array(img2)
        #
        # # pass in to input_transform because it was in the original repo
        # img1 = input_transform(img1)
        # img2 = input_transform(img2)
        #=========================

        img1 = input_transform(imread(img1_file))
        img2 = input_transform(imread(img2_file))
        input_var = torch.cat([img1, img2]).unsqueeze(0)

        if args.bidirectional:
            # feed inverted pair along with normal pair
            inverted_input_var = torch.cat([img2, img1]).unsqueeze(0)
            input_var = torch.cat([input_var, inverted_input_var])

        input_var = input_var.to(device)
        # compute output
        output = model(input_var)
        if args.upsampling is not None:
            output = F.interpolate(output, size=img1.size()[-2:], mode=args.upsampling, align_corners=False)
        for suffix, flow_output in zip(['flow', 'inv_flow'], output):
            # NOTE made this img2 as the position is relative to that img
            filename = save_path/'{}{}'.format(img2_file.stem, suffix)
            if args.output_value in['vis', 'both']:
                rgb_flow = flow2rgb(args.div_flow * flow_output, max_value=args.max_flow)
                to_save = (rgb_flow * 255).astype(np.uint8).transpose(1,2,0)
                imwrite(filename + '.png', to_save)
            if args.output_value in ['raw', 'both']:
                # print('saving npy file to: ', filename)
                # Make the flow map a HxWx2 array as in .flo files
                to_save = (args.div_flow*flow_output).cpu().numpy().transpose(1,2,0)
                np.save(filename + '.npy', to_save)
                # print('\n\n\nFlow is: ', to_save.shape)


if __name__ == '__main__':
    main()
