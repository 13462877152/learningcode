################################################################################
# MC-GAN
# Modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# By Samaneh Azadi
################################################################################

import time
from options.train_options import TrainOptions
opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

from models.models import create_model
from util.visualizer import Visualizer
from data.data_loader import CreateDataLoader

opt.stack = True
data_loader = CreateDataLoader(opt)

dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)
opt.use_dropout = False  ##don't use dropout for the generator in glyphnet
opt.use_dropout1 = True  #use   dropout for the generator in ornament
model = create_model(opt)
visualizer = Visualizer(opt)

total_steps = 0
  
epoch =int(opt.which_epoch1) #load from which epoch
epoch0 = epoch 
print("starting propagating back to the first network with starting lr %s ..."%opt.lr)
opt.lr = opt.lr
opt.continue_train = False
opt.use_dropout = True  #use dropout for the generator in glyphnet,什么时候用，什么时候不用呢？
opt.use_dropout1 = True #use  dropout for the generator in ornament,
model = create_model(opt)
visualizer = Visualizer(opt) 
print('saving the model at the end of epoch %d, iters %d' %
    (epoch0, total_steps))
model.save(epoch0)

for epoch in range(1, opt.niter + opt.niter_decay + 1):#（1到400+300）
    epoch_start_time = time.time()
    for i, data in enumerate(dataset):#data 返回后包含A， A_base, A_paths, B, B_paths五部分，A：torch.Size([4, 26, 64, 64]) A_paths：四张路片的路径， A_base：torch.Size([1, 26, 64, 64])
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter = total_steps - dataset_size * (epoch - 1)
        model. set_input(data)
        if not opt.no_Style2Glyph:  #opt.no_Style2Glyph,表示不执行到glyphnet的反向传播，not则相反，表示执行从ornament到glyphnet的反向传播
            model.optimize_parameters_Stacked(epoch)  #表示执行从ornament到glyphnet的反向传播,也就是还要执行backward_G,端到端训练时会这样执行
        else:
            model.optimize_parameters(epoch)  #表示不执行到glyphnet的反向传播，当执行train_StackGAN.sh中的第二部分时会选择这个。

    #这个if语句用来显示结果，显示的内容在model.get_current_visuals()确定，如何显示有display_current_results来选择是用html还是用别的
        if total_steps % opt.display_freq == 0:
            visualizer.display_current_results(model.get_current_visuals(), epoch)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()#以字典形式返回各个损失函数
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch+epoch0, total_steps))
            model.save('latest')

    if (epoch % opt.save_epoch_freq == 0):# or (epoch<20):
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch+epoch0, total_steps))
        model.save('latest')
        model.save(epoch+epoch0)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    if epoch > opt.niter:
        model.update_learning_rate()
