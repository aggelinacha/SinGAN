import os
from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.imresize import imresize, imresize_to_shape
import SinGAN.functions as functions


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--label_dir', help='input label dir', default='Input/Images')
    parser.add_argument('--input_label', help='input label name', required=True)
    parser.add_argument('--mode', default='random_samples')
    parser.add_argument('--gen_start_scale', type=int, help='generation start scale', default=0)
    parser.add_argument('--scale_h', type=float, help='horizontal resize factor for random samples', default=1.5)
    parser.add_argument('--scale_v', type=float, help='vertical resize factor for random samples', default=1)
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)
    if dir2save is None:
        print('task does not exist')
    elif (os.path.exists(dir2save)):
        print('already exists')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        real_ = functions.read_image(opt)
        functions.adjust_scales2image(real_, opt)
        real = imresize(real_, opt.scale1, opt)
        reals = functions.creat_reals_pyramid(real, [], opt)
        Gs, Zs, _, NoiseAmp = functions.load_trained_pyramid(opt)
        in_s = functions.generate_in2coarsest(reals, 1, 1, opt)
        # Load labels
        label_paths = sorted(os.listdir(opt.label_dir))
        for l in label_paths:
            print(l)
            opt.input_label = l
            label_ = functions.read_label(opt)
            label = imresize_to_shape(label_, real.shape[2:], opt)
            labels = functions.creat_reals_pyramid(label, [], opt)
            SinGAN_generate(Gs, Zs, reals, labels, NoiseAmp, opt,
                            gen_start_scale=opt.gen_start_scale,
                            num_samples=1)
