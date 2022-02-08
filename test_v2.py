import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import util, html, imresize
from util import html
import numpy as np
import os
import sys
import ntpath
import time


# save image to the disk
def save_images(webpage, visuals, image_path, output_dir, aspect_ratio=1.0, width=256, f_name="", citysc=False):
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]
    # content ve stil testlerini yapmak icin ben ekledim!!!
    aaa = os.path.basename(os.path.dirname(image_path[0]))
    if not os.path.exists(os.path.join(image_dir, aaa)):
        os.makedirs(os.path.join(image_dir, aaa))

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        # if label == "real_A":
        #     continue

        # image_name = '%s.png' % (name)
        # image_name = f_name # cityscape icin eklendi
        if citysc:
            im = imresize(im, (1024, 2048))  # cityscape icin eklendi
            image_name = os.path.splitext(f_name)[0] + ".png"  # cityscape icin eklendi
        else:
            image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        h, w, _ = im.shape
        if aspect_ratio > 1.0:
            im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
        if aspect_ratio < 1.0:
            im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
        util.save_image(im, save_path)
        util.save_image(im, os.path.join(output_dir, image_name))

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


if __name__ == '__main__':
    opt = TestOptions().parse()
    # hard-code some parameters for test
    opt.num_threads = 1  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    model.setup(opt)
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # pix2pix: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # CycleGAN: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    # added for cityscapes.
    if opt.cityscapes:
        with open(opt.cityscape_fnames) as f:
            f_names = f.read().split('\n')

    # output_path
    if not os.path.exists(opt.results_dir):
        os.makedirs(opt.results_dir, exist_ok=True)

    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        if opt.cityscapes:
            index = int(os.path.basename(img_path[0]).split("_")[0]) - 1  # cityscapes
        if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))
        if not opt.cityscapes:
            save_images(webpage, visuals, img_path, opt.results_dir, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, citysc=False)
        else:
            save_images(webpage, visuals, img_path, opt.results_dir, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize,
                        f_name=f_names[index], citysc=True)  # cityscapes
    # save the website
    webpage.save()
