################################################################################
# MC-GAN
# Modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# By Samaneh Azadi
#程序验证输入./datasets/public_web_fonts/BLOOD/A/test/BLOOD.png',（self.input_A0，只有几个字母）先输入glyphnet ,后生成了./results/BLOOD_MCGAN_train/test_400+700/images/BLOOD_(real_A.png,此时是个黑色的26个字母排列的图片)，然后在把这个图片输入ornament网络生成器，生成了results/BLOOD_MCGAN_train/test_400+700/images/BLOOD_fake_B.png，(此时是26个和前面少数字母风格一致的图片），results/BLOOD_MCGAN_train/test_400+700/images/BLOOD_real_B.png是真的图片，二者可以对比一下。
################################################################################


import time
import os
from options.test_options import TestOptions
opt = TestOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html

opt.nThreads = 1   # test code only supports nThreads=1
opt.batchSize = 1  #test code only supports batchSize=1
opt.serial_batches = True # no shuffle 测试时不打乱顺序？
opt.stack = True
opt.use_dropout = False # 测试时不用
opt.use_dropout1 = False # 测试时不用



data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)

# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch+'+'+opt.which_epoch1))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch+'+'+opt.which_epoch1))
# test
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)#该函数用来把for循环中data的A,B,B_paths分别赋值给self中对应变量self.input_A0,self.input_BO,self.image_paths，
    model.test()#把self.input_A0（1,26,64,64）通过G1生成图片self.fake_B0 (1,26,64,64) ，端到端测试时，利用fake_B0 构造出real_A1（26,3,64,64），variable封装后赋值给self_real_A1（26,3,64,64），然后通过ornament 网络生成 self.fake_B1,把input_B0经variable封装后赋值给self_real_B1
    visuals = model.get_current_visuals()#visuals 包含‘fake_B',’real_A','real_B'三部分，shape都为（64,1664,3）,是把self.fake_B1.data,self.real_A1.data,self.real_B1.data三部分处理后变过来的。
    img_path = model.get_image_paths() # 此处是input_B0（self.real_B1）的路径'./datasets/public_web_fonts/BLOOD/B/test/BLOOD.png',)
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()