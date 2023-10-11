import torch
import math
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import math
import kornia as K
from matplotlib import pyplot as plt

class PoissonCompositionLayer(torch.nn.Module):
    # def __init__(self):
    #     """
    #     In the constructor we instantiate four parameters and assign them as
    #     member parameters.
    #     """
    #     super().__init__()
    #     self.a = torch.nn.Parameter(torch.randn(()))
    #     self.b = torch.nn.Parameter(torch.randn(()))
    #     self.c = torch.nn.Parameter(torch.randn(()))
    #     self.d = torch.nn.Parameter(torch.randn(()))
    # def poisson_comp(target, source, mask):

    def forward(self, target, source, mask):
        # target: target background BxCxHxW: [0~255]
        # source: source background BxCxHxW: [0~255]
        # mask: source background BxCxHxW: [0~1]
        # an update to accelerate computation (only composite mask region, this can make the computation be approximately independent of image size)
        

        # get input iamge information
        H = target.shape[-2]
        W = target.shape[-1]
        # define and centralize the frequency
        [wx, wy] = torch.meshgrid( torch.arange(1,2*W+1), torch.arange(1,2*H+1) )
        [wx, wy] = torch.meshgrid( torch.arange(1,2*W+1), torch.arange(1,2*H+1) )
        wx = wx.T
        wy = wy.T
        wx0 = W+1
        wy0 = H+1
        wx = wx - wx0 
        wy = wy - wy0 
        grad_x_t, grad_y_t = self.grad_img(target)
        grad_x_s, grad_y_s = self.grad_img(source)
        grad_x_V = (1-mask)* grad_x_t + mask * grad_x_s
        grad_y_V = (1-mask)* grad_y_t + mask * grad_y_s
        # Reflection padding to achieve periodical boundary
        grad_x_V_pdd = torch.cat((grad_x_V, -torch.flip(grad_x_V, (3,))),3)
        grad_x_V_pdd = torch.cat((grad_x_V_pdd, torch.flip(grad_x_V_pdd, (2,))),2)

        grad_y_V_pdd = torch.cat((grad_y_V, -torch.flip(grad_y_V, (2,))),2)
        grad_y_V_pdd = torch.cat((grad_y_V_pdd, torch.flip(grad_y_V_pdd, (3,))),3)

        # Calculate 2 dimensional fft of input
        dft_x = torch.fft.fftshift(torch.fft.fft2(grad_x_V_pdd),[-1,-2])
        dft_y = torch.fft.fftshift(torch.fft.fft2(grad_y_V_pdd),[-1,-2])
        coefficients_dft = ((2*math.pi*1j*wx/(2*W))*dft_x + (2*math.pi*1j*wy/(2*H))*dft_y)/((2*math.pi*1j*wx/(2*W))**2+(2*math.pi*1j*wy/(2*H))**2+1e-10)
        coefficients_dft[:,:, int(wy0)-1, int(wx0)-1] = 0
        # recover the image in real domain 
        u = torch.fft.ifft2(torch.fft.fftshift(coefficients_dft,[-1,-2]))
        res = u.real[...,:H,:W]
        # caliberate the recovered result to have the same mean value
        mean1 = torch.sum(((target)*(1-mask)),[-1,-2])/torch.sum((1-mask),[-1,-2])
        mean2 = torch.sum((res*(1-mask)),[-1,-2])/torch.sum((1-mask),[-1,-2])
        res1 = torch.clone(res)
        for i in range(3):
            res1[:,i,:,:] = res1[:,i,:,:]  + mean1[0,i]- mean2[0,i]
        return res1
    
    def grad_img(self, input):
        # calculate the hight and width of the image
        H = input.shape[-2]
        W = input.shape[-1]
        # define the frequency
        [wx, wy] = torch.meshgrid( torch.arange(1,2*W+1), torch.arange(1,2*H+1) )
        wx = wx.T
        wy = wy.T
        # centralize the frequency
        wx0 = W+1
        wy0 = H+1
        wx = wx - wx0 
        wy = wy - wy0 
        # Reflection padding to achieve periodical boundary
        input_ = torch.cat((input, torch.flip(input, (3,))),3)
        input_ = torch.cat((input_, torch.flip(input_, (2,))),2)
        # Calculate 2 dimensional fft of input
        dft = torch.fft.fftshift(torch.fft.fft2(input_),[-1,-2])
        # compute in the fourier domain
        gradx_fourier = (2*math.pi*1j/(2*W)*wx)*dft
        grady_fourier = (2*math.pi*1j/(2*H)*wy)*dft
        # recover the real domain gradient
        gradx_color = torch.fft.ifft2(torch.fft.fftshift(gradx_fourier,[-1,-2])).real
        grady_color = torch.fft.ifft2(torch.fft.fftshift(grady_fourier,[-1,-2])).real
        
        return gradx_color[:,:,:H,:W], grady_color[:,:,:H,:W]