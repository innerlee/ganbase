from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os

from ganbase.evaluation.inception import InceptionV3
from ganbase.evaluation.inception_score import calculate_is_given_path

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', type=str, help='Path to image dir')
    parser.add_argument('--nsamples', type=int, help='The number of randomly selected samples')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size to use')
    parser.add_argument('--splits', type=int, default=10, help='Splits to cal')
    parser.add_argument('--dims', type=int, default=2048,
                        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                        help=('Dimensionality of Inception features to use. '
                              'By default, uses pool3 features'))
    parser.add_argument('--gpu', default='', type=str, help='GPU to use (leave blank for CPU only)')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    is_m, is_s = calculate_is_given_path(args.path,
                                         args.nsamples,
                                         args.batch_size,
                                         args.splits,
                                         args.gpu != '',
                                         args.dims)
    print(is_m, is_s)
