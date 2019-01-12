from PIL import Image
import torch
from torchvision.transforms import ToTensor, ToPILImage

path = '../Users/rmahfuz/Desktop/bme_hw/'

class Conv2D:
    def __init__(self, in_channel, o_channel, kernel_size, stride, mode):
        self.in_channel = in_channel
        self.o_channel = o_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.mode = mode
        self.K1 = torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=torch.float32)
        self.K2 = torch.tensor([[-1,  0,  1], [-1, 0, 1], [-1, 0, 1]], dtype=torch.float32)
        self.K3 = torch.tensor([[1,  1,  1],[1, 1, 1], [1, 1, 1]], dtype=torch.float32)
        self.K4 = torch.tensor([[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=torch.float32)
        self.K5 = torch.tensor([[-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1]], dtype=torch.float32)
    def forward(self, in_img):
        '''in_img is a 3d float tensor
        returns [num_of_ops, 3d float tensor]'''
        ops = 0
        #-------------------------------------------------------------------------
        if self.mode == 'known' and self.o_channel == 1: #task 1
            tmp, out0 = calc(in_img[0], self.K1, self.stride); ops += tmp
            tmp, out1 = calc(in_img[1], self.K1, self.stride); ops += tmp
            tmp, out2 = calc(in_img[2], self.K1, self.stride); ops += tmp
            out = (out0+out1+out2)/3 #3d tensor
            ops += (in_img.shape[1] * in_img.shape[2]*2)
            out = out.reshape(1, out.shape[0], out.shape[1])
            return [ops, out]
        #-------------------------------------------------------------------------
        if self.mode == 'known' and self.o_channel == 2 and self.kernel_size == 5: #task 2
            tmp, out0 = calc(in_img[0], self.K4, self.stride); ops += tmp
            tmp, out1 = calc(in_img[1], self.K4, self.stride); ops += tmp
            tmp, out2 = calc(in_img[2], self.K4, self.stride); ops += tmp
            out_first = (out0+out1+out2)/3 #3d tensor
            ops += (in_img.shape[1] * in_img.shape[2]*2)
            #out_first = out.reshape(1, out.shape[0], out.shape[1])
            #-----------------------------------------------------
            tmp, out0 = calc(in_img[0], self.K5, self.stride); ops += tmp
            tmp, out1 = calc(in_img[1], self.K5, self.stride); ops += tmp
            tmp, out2 = calc(in_img[2], self.K5, self.stride); ops += tmp
            out_second = (out0+out1+out2)/3 #3d tensor
            ops += (in_img.shape[1] * in_img.shape[2]*2)
            #out_second = out.reshape(1, out.shape[0], out.shape[1])
            #-----------------------------------------------------
            out = torch.stack([out_first, out_second])
            return [ops, out]
        #-------------------------------------------------------------------------
        if self.mode == 'known' and self.o_channel == 3 and self.kernel_size == 3:
            tmp, out0 = calc(in_img[0], self.K1, self.stride); ops += tmp
            tmp, out1 = calc(in_img[1], self.K1, self.stride); ops += tmp
            tmp, out2 = calc(in_img[2], self.K1, self.stride); ops += tmp
            out_first = (out0+out1+out2)/3 #3d tensor
            ops += (in_img.shape[1] * in_img.shape[2]*2)
            #-----------------------------------------------------
            tmp, out0 = calc(in_img[0], self.K2, self.stride); ops += tmp
            tmp, out1 = calc(in_img[1], self.K2, self.stride); ops += tmp
            tmp, out2 = calc(in_img[2], self.K2, self.stride); ops += tmp
            out_second = (out0+out1+out2)/3 #3d tensor
            ops += (in_img.shape[1] * in_img.shape[2]*2)
            #-----------------------------------------------------
            tmp, out0 = calc(in_img[0], self.K3, self.stride); ops += tmp
            tmp, out1 = calc(in_img[1], self.K3, self.stride); ops += tmp
            tmp, out2 = calc(in_img[2], self.K3, self.stride); ops += tmp
            out_third = (out0+out1+out2)/3 #3d tensor
            ops += (in_img.shape[1] * in_img.shape[2]*2)
            #-----------------------------------------------------
            out = torch.stack([out_first, out_second, out_third])
            return [ops, out]
        #-------------------------------------------------------------------------
        if self.mode == 'rand': #part b and c
            li = []; ops = 0
            for i in range(0, self.o_channel):
                kernel = torch.rand(self.kernel_size, self.kernel_size)
                tmp, out0 = calc(in_img[0], kernel, self.stride); ops += tmp
                tmp, out1 = calc(in_img[1], kernel, self.stride); ops += tmp
                tmp, out2 = calc(in_img[2], kernel, self.stride); ops += tmp
                li.append((out0+out1+out2)/3) #3d tensor
                ops += (in_img.shape[1] * in_img.shape[2]*2)
            out = torch.stack(li)
            return [ops, out]
                


#kernel definitions:
K1 = torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=torch.float32)
'''K2 = torch.tensor([[-1,  0,  1], [-1, 0, 1], [-1, 0, 1]])
K3 = torch.tensor([[1,  1,  1],[1, 1, 1], [1, 1, 1]])
K4 = torch.tensor([[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
K5 = torch.tensor([[-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1]])'''

def calc(img, kernel, stride): #img should be a 2d tensor
    out = []; ops = 0
    for i in range(0, img.shape[0] - kernel.shape[0], stride): # go down
        row = []
        for j in range(0, img.shape[1] - kernel.shape[1], stride): #go right
            ans = 0
            for k in range(0, kernel.shape[0]):
                for l in range(0, kernel.shape[1]):
                    ans += kernel[k][l] * img[i+k][j+l]
            #ops += (kernel.shape[0] * kernel.shape[1]) + 1 #multiplications + addition
            ops += ((kernel.shape[0] * kernel.shape[1]) *2) -1 #multiplications + addition
            row.append(ans)
        out.append(row)
    return [ops, torch.tensor(out)]

'''#import images
tensor = ToTensor()
img1 = tensor(Image.open(path + "example.png")) #size 1280x720
out = calc(img1[0], K1, 1)
print(out.shape)
out = out.reshape(1, out.shape[0], out.shape[1])
print(out.shape)
to_img = ToPILImage()
to_img(out).save(path + 'out_example.png')'''

    
