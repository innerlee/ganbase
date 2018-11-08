from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os

from ganbase.evaluation.inception import InceptionV3
from ganbase.evaluation.fid_distance import  calculate_fid_given_paths


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', type=str, nargs=2,
                        help=('Path to the generated images or '
                              'to .npz statistic files'))
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size to use')
    parser.add_argument('--dims', type=int, default=2048,
                        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                        help=('Dimensionality of Inception features to use. '
                              'By default, uses pool3 features'))
    parser.add_argument('--gpu', default='', type=str,
                        help='GPU to use (leave blank for CPU only)')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    fid_value = calculate_fid_given_paths(args.path, args.batch_size, args.gpu != '', args.dims)
    print('FID: ', fid_value)