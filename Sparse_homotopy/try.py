from importlib.metadata import requires
import time
from parsers import BaseParser
import os

from datasets import BaseDataset
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import numpy as np

from torchvision.transforms import ToPILImage
import torch.backends.cudnn as cudnn



import inception_v3, resnet



def G(original_img, original_label, delta, epsilon, net):
    
    
    itr = 1
    norm = torch.norm(delta)
    # print("norm:", norm.item())
    delta1 = delta.clone().detach() / norm
    
    start, end = 0.0, norm.item()
    
    logist = net(original_img.data + end * delta1)
    _, target = torch.max(logist,1)
    
    if target == original_label:
        print("="*50, "problem")
        exit(0)
    
    
    while True:
        # print("Iteration:", itr, start, end)
        itr += 1
        
        if abs(end - start) < 1e-4:
            return end
        
        mid = (start + end) / 2
        
        logist = net(original_img.data + mid * delta1)
        _, target = torch.max(logist,1)
        
        if target.item() == original_label:
            start = mid 
        else:
            end = mid
    
    
    ########### targetted
    
    # for e in [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]:
    #     logist = net(original_img.data + e * delta)
    #     _, target = torch.max(logist,1)
        
    #     print("Predicted label: ", target.item(), "for e =", e)
    
import copy
        
def gradient(original_img, original_label, delta, epsilon, net):
    
    h = 5e-5
    
    grad = torch.zeros_like(delta)
    print("sizeeeeee", delta.shape)
    
    for i in range(delta.shape[1]):
        print("channel", i, "starting")
        for j in range(delta.shape[2]):
            for k in range(delta.shape[3]):
                # print(i, j, k)
                
                # print(delta[0, i, j, k])
                delta_high = delta.clone().detach()
                # print(delta_high[0, i, j, k])
                delta_high[0, i, j, k] += h 
                # print(delta_high[0, i, j, k], delta[0, i, j, k])
                
                # print("test", torch.norm(delta_high - delta))
                
                delta_low = delta.clone().detach()
                delta_low[0, i, j, k] -= h
                
                high = G(original_img, original_label, delta_high, epsilon, net)
                low =  G(original_img, original_label, delta_low, epsilon, net)
                # print(high, low)
                # if abs(high - low) > 1e-7:
                #     print("huaaaa", abs(high - low)/(2*h))
                
                grad[0, i, j, k] = (high - low) / (2 * h)
                
    return grad

from tqdm import tqdm
def gradient_approx(original_img, original_label, delta, epsilon, net, func_val, Q=2000):
    
    beta = 0.005
    grads = torch.zeros_like(delta)
    
    for q in tqdm(range(Q)):
        # print("gaya")
        # print("number", q)
        u = torch.randn_like(delta).cuda() * 0.5
        # print(u.min(), u.max())
        
        high = G(original_img, original_label, delta + beta * u, epsilon, net)
        # print("high", high - func_val)
        grads += ((high - func_val) / beta) * u
    grads /= Q
    return grads


def main():
    print("start....")
    
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    
    
    # net = inception_v3.inception_v3(pretrained=False)
    # net.load_state_dict(torch.load('./pretrain/inception_v3_google-1a9a5a14.pth'))
    # net.eval().cuda()


    net = resnet.ResNet18()
    net = net.to('cuda')
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    checkpoint = torch.load('./pretrain/resnet.pth')
    net.load_state_dict(checkpoint['net'])
    net.eval()
    # net.load_state_dict(torch.load(''))
    
    
    # test_dataset = BaseDataset(
    #     "imgs",
    #     transform=transforms.Compose([
    #         transforms.Resize((299, 299)),
    #         transforms.ToTensor()
    #     ]),
    # )
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    test_dataset = BaseDataset(
        "cifar_10_imgs",
        transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    
    print("loader created")
    
    for idx, data in enumerate(test_loader):

        input = data['img']
        input = input.cuda()
        ori_image = Variable(input,requires_grad = False)
        
        logist = net(ori_image.data)
        _, target = torch.max(logist,1)
    
        print("Predicted label for original image: ", target.item())
    
        # print(ori_image)
        
        delta = Variable(torch.rand_like(ori_image), requires_grad=False).cuda()
        # print("-" * 20)
        # print(delta)
        val = G(ori_image, target.item(), delta, 1e-3, net)
        print("val: ", val)
        # val.backward()
        # print(delta.grad)

        import time
        start = time.time()
        grad = gradient(ori_image, target.item(), delta, 1e-3, net)
        end = time.time()
        
        print(grad.min(), grad.max())
        print("Time for gradient:", end - start)
        torch.save(grad, "actual_grad1.pth")


        start = time.time()
        grad_approx = gradient_approx(ori_image, target.item(), delta, 1e-3, net, val)
        end = time.time()
        
        print("Time for approximate gradient:", end - start)
        
        print(grad_approx.min(), grad_approx.max())
        torch.save(grad_approx, "approx_grad.pth")
        print("distance:", torch.norm(grad - grad_approx))

        
        exit(0)


        
        # start = time.time()
        # grad = gradient_approx(ori_image, target.item(), delta, 1e-3, net, val)
        # print(grad.min(), grad.max())
        # end = time.time()
        
        # print("valll", G(ori_image, target.item(), delta, 1e-3, net))        

if __name__ == "__main__":
    main()

