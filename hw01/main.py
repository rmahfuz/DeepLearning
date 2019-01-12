from conv import Conv2D
from PIL import Image
import torch
import timeit 
from torchvision.transforms import ToTensor, ToPILImage

path = '../Users/rmahfuz/Desktop/bme_hw/'
tensor = ToTensor()
to_img = ToPILImage()

#import images
#img1 = tensor(Image.open(path + "small.jpg")) #size 1280x720
img2 = tensor(Image.open(path + "big.jpg")) #size 1920x1080

#----------------------------part a---------------------------------
#---------task 1----------
conv2d = Conv2D(3, 1, 3, 1, 'known')
start = timeit.default_timer()
ops, out_img = conv2d.forward(img2)
stop = timeit.default_timer()
print('Time: ', stop - start) 
#print('out_img.shape = ', out_img.shape)
print('ops: ', ops)
to_img(out_img).save(path + 'out_big_k1.png') #save out_img

#---------task 2----------
conv2d = Conv2D(3, 2, 5, 1, 'known')
start = timeit.default_timer()
ops, out_img = conv2d.forward(img2)
stop = timeit.default_timer()
print('Time: ', stop - start) 
#print('out_img.shape = ', out_img.shape)
print('ops: ', ops)
tmp0 = out_img[0]; tmp0 = tmp0.resize(1, tmp0.shape[0], tmp0.shape[1])
to_img(tmp0).save(path + 'out_big_k4.png') #save out_img
tmp1 = out_img[1]; tmp1 = tmp1.resize(1, tmp1.shape[0], tmp1.shape[1])
to_img(tmp1).save(path + 'out_big_k5.png') #save out_img


#---------task 3----------
conv2d = Conv2D(3, 3, 3, 2, 'known')
start = timeit.default_timer()
ops, out_img = conv2d.forward(img2)
stop = timeit.default_timer()
print('Time: ', stop - start) 
#print('out_img.shape = ', out_img.shape)
print('ops: ', ops)
tmp0 = out_img[0]; tmp0 = tmp0.resize(1, tmp0.shape[0], tmp0.shape[1])
to_img(tmp0).save(path + 'out_big_k1_stride2.png') #save out_img
tmp1 = out_img[1]; tmp1 = tmp1.resize(1, tmp1.shape[0], tmp1.shape[1])
to_img(tmp1).save(path + 'out_big_k2_stride2.png') #save out_img
tmp2 = out_img[2]; tmp2 = tmp2.resize(1, tmp2.shape[0], tmp2.shape[1])
to_img(tmp2).save(path + 'out_big_k3_stride2.png') #save out_img


#-------------------part b (varying o_channel)-------------------
'''time_taken = [0]*11
for i in range(0, 11):
    conv2d = Conv2D(3, 2**i, 3, 1, 'rand')
    start = timeit.default_timer()
    ops, out_img = conv2d.forward(img1)
    stop = timeit.default_timer()
    time_taken[i] = stop - start
print(time_taken)
#plot i vs. time_taken

#-------------------------
i = 4
conv2d = Conv2D(3, 2**i, 3, 1, 'rand')
start = timeit.default_timer()
ops, out_img = conv2d.forward(img1)
stop = timeit.default_timer()
print('for i = ', i, ', time taken: ',stop - start)'''
#-------------------part c (varying kernel_size)------------------------
'''ops = [0]*5
for j in [3,5,7,9,11]:
    conv2d = Conv2D(3, 2, j, 1, 'rand')
    ops[j], out_img = conv2d.forward(img1)
#plot ops vs. [3,5,7,9,11]

j = 7
conv2d = Conv2D(3, 2, j, 1, 'rand')
start = timeit.default_timer()
ops, out_img = conv2d.forward(img1)
stop = timeit.default_timer()
print('time taken for i = ', j, ' = ', stop-start, 'ops = ', ops)
'''

