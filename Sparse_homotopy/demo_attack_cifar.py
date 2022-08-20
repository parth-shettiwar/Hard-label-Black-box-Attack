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



import inception_v3
import resnet

import debugpy


# debugpy.listen(5565)
# print("Waiting for debugger")
# debugpy.wait_for_client()
# print("Attached! :)")

GRAD_EVALUATIONS = 0


cudnn.benchmark = True 

def after_attack(x, net, original_img, original_class, target_class, post, loss_type, tar, iters, val_w1, val_w2, max_epsilon):
    
    global GRAD_EVALUATIONS

    if post == 1:
        s1 = 1e-3
        s2 = 1e-4
        max_iter = 40000
    else:
        s1 = val_w2
        s2 = val_w1
        max_iter = iters

    mask = torch.where(torch.abs(x.data) > 0, torch.ones(1).cuda(), torch.zeros(1).cuda())

    logist = net(x.data+original_img.data)
    _,target=torch.max(logist,1)


    pre_x = x.data

    for i in range(max_iter):

        temp = Variable(x.data, requires_grad=True)
        logist = net(temp + original_img.data)
        if tar == 1:
            if loss_type == 'ce':
                ce = torch.nn.CrossEntropyLoss()
                Loss = ce(logist,torch.ones(1).long().cuda()*target_class)
            elif loss_type == 'cw':
                Loss = CWLoss(logist, torch.ones(1).long().cuda()*target_class, kappa=0, tar = True)
        else:
            Loss = CWLoss(logist, torch.ones(1).long().cuda()*target_class, kappa=0, tar = False)

        net.zero_grad()
        if temp.grad is not None:
            temp.grad.data.fill_(0)
        Loss.backward()
        grad = temp.grad
        
        GRAD_EVALUATIONS += 1

    
        temp2 = Variable(x.data, requires_grad=True)
        Loss2 = torch.norm(temp2, p=float("inf"))
        net.zero_grad()
        if temp2.grad is not None:
            temp2.grad.data.fill_(0)
        Loss2.backward()
        grad2 = temp2.grad
        
        GRAD_EVALUATIONS += 1


        pre_x = x.data

        pre_noise = temp2.data
        if post == 0:
            temp2 = temp2.data - s1*grad2.data*mask - s2*grad.data*mask
        else:
            temp2 = temp2.data - s1*grad2.data*mask

        thres = max_epsilon
        temp2 = torch.clamp(temp2.data, -thres, thres)
        temp2 = torch.clamp(original_img.data+temp2.data, 0, 1)
    
        x = temp2.data - original_img.data
    

        logist = net(x.data + original_img.data)
        _,pre=torch.max(logist,1)
        if(post == 1):
            if tar ==  1:
                if(pre.item() != target_class):
                    success = 1
                    return pre_x
                    break
            else:
                if(pre.item() == target_class):
                    success = 1
                    return pre_x
                    break

    return x

def F(x, loss_type, net, lambda1, original_img, target_class, tar):
    temp = Variable(x.data, requires_grad=False)
    logist = net(temp+original_img.data)
    if tar == 1:
        if loss_type == 'ce':
            ce = torch.nn.CrossEntropyLoss()
            Loss = ce(logist,torch.ones(1).long().cuda()*target_class)
        elif loss_type == 'cw':
            Loss = CWLoss(logist, torch.ones(1).long().cuda()*target_class, kappa=0, tar = True)
    else:
        Loss = CWLoss(logist, torch.ones(1).long().cuda()*target_class, kappa=0, tar = False)
    res = Loss.item() + lambda1*torch.norm(x.data,0).item()
    net.zero_grad()
    return res


def G_loss(x, original_img, original_label, net):
    
    itr = 1
    norm = torch.norm(x.data)
    # print("L2 norm: ", norm)
    if norm < 1e-8:
        # logist = net(x.data + original_img.data)
        # Loss = CWLoss(logist, torch.ones(1).long().cuda()*original_label, kappa=0, tar = False)
        # print("Finding normal loss:", Loss)
        # return Loss
        return 1000 # some high value
        
    delta1 = x.data.clone().detach() / norm
    # print(original_img.min(), original_img.max())
    
    step_size = 5
    i = 0
    start, end = -1, -1
    while True:
        
        temp_end = (i + 1) * step_size * norm.item()
        
        # img = torch.clamp(original_img.data + temp_end * delta1, 0, 1)
        logist = net(original_img.data + temp_end * delta1)
        # logist = net(img)
        _, target = torch.max(logist,1)
        
        # print("loop", i, "norm", norm.item(), "target", target)
        if target != original_label:
            # print("end is ", i)
            start = i * step_size * norm.item()
            end = (i + 1) * step_size * norm.item()
            break
        
        if i > 1000:
            # print("L0 norm: ", torch.norm(x, 0))
            return 1001
        i += 1
        
    
    # start, end = 0.0, 50 * norm.item()
    
    logist = net(original_img.data + end * delta1)
    _, target = torch.max(logist,1)
    
    # print("norm:", norm.item(), original_label, target)
    
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
            
def G(x, original_img, original_label, net, lambda1):
    
    net.eval().cuda()
    Loss = G_loss(x, original_img, original_label, net)
    res = Loss + lambda1*torch.norm(x.data,0).item()
    net.zero_grad()
    # print("G evaluated")
    return res
    
def G_grad(x, original_img, original_label, net, Q=20):
    
    beta = 0.05
    grads = torch.zeros_like(x.data)
    func_val = G_loss(x, original_img, original_label, net)
    
    for q in range(Q):
        u = torch.randn_like(x).cuda() * 0.5
        # print(u.min(), u.max())
        high = G_loss(x + beta * u, original_img, original_label, net)
        # print("high", high - func_val)
        grads += ((high - func_val) / beta) * u
    grads /= Q
    return grads




def prox_pixel(x, alpha, lambda1, original_img, max_epsilon):

    temp_x = x.data * torch.ones(x.shape).cuda()

    thres = max_epsilon
    clamp_x = torch.clamp(temp_x.data, -thres, thres)

    temp_img = original_img.data + clamp_x.data
    temp_img = torch.clamp(temp_img.data, 0, 1)
    clamp_x = temp_img.data - original_img.data

    val = 1 / (2*alpha*lambda1)
    cond = 1 + val * (clamp_x-temp_x)*(clamp_x-temp_x) > val * temp_x*temp_x
    cond = cond.cuda()

    res = torch.zeros(x.shape).cuda()
    res = torch.where(cond, res, clamp_x.data)
    return res



def nmAPG(x0, loss_type, net, eta, delta, rho, original_img, lambda1, search_lambda_inc, search_lambda_dec, target_class, original_class, tar, max_update, maxiter, max_epsilon):

    global GRAD_EVALUATIONS
    
    x0_norm0 = torch.norm(torch.ones(x0.shape).cuda()*x0.data,0).item()
    max_update = max_update

    Loss = G_loss(x0, original_img, original_class, net)

    z = x0
    y_pre = torch.zeros(original_img.shape).cuda()

    pre_loss = 0
    cur_loss = 0

    counter = 0
    success = 0

    alpha_y = 1e-3
    alpha_x = 1e-3

    alpha_min = 1e-20
    alpha_max = 1e20
    x_pre = x0
    x = x0
    t = 1
    t_pre = 0
    c = Loss + lambda1*torch.norm(x.data,0)
    q = 1
    k = 0
    while True:
        print("nmAPG iter: ", k, torch.norm(x))
        y = x + t_pre/t*(z-x) + (t_pre-1)/t*(x-x_pre)

        if k > 0:
            s = y - y_pre.data
            
            print("y norm: ", torch.norm(y))
            grad_y = G_grad(y, original_img, original_class, net)
            
            GRAD_EVALUATIONS += 1

            #gradient of yk-1
            
            grad_y_pre = G_grad(y_pre, original_img, original_class, net)

            
            GRAD_EVALUATIONS += 1

            r = grad_y - grad_y_pre
            
            #prevent error caused by numerical inaccuracy
            if torch.norm(s,1) < 1e-5:
                s = torch.ones(1).cuda()*1e-5
            
            if torch.norm(r,1) < 1e-10:
                r = torch.ones(1).cuda()*1e-10

            alpha_y = torch.sum(s*r)/torch.sum(r*r)
            alpha_y = alpha_y.item()
        
        temp_alpha = alpha_y

        if temp_alpha < alpha_min:
            temp_alpha = alpha_min

        if temp_alpha > alpha_max:
            temp_alpha = alpha_max

        if np.isnan(temp_alpha):
            temp_alpha = alpha_min
        alpha_y = temp_alpha

        count1 = 0
        while True:
            print("Inner: ", count1)
            count1 = count1 + 1
            if count1 > 1000:
                break
            
            grad_y = G_grad(y, original_img, original_class, net)
            # print(grad_y)
            
            GRAD_EVALUATIONS += 1

            z = prox_pixel(x=y-alpha_y*grad_y,alpha=alpha_y,lambda1=lambda1,original_img=original_img, max_epsilon=max_epsilon)
            # print(z)
            #increase lambda
            if(search_lambda_inc == 1):
                if(torch.norm(z,1) != 0):
                    return 0
                else:
                    return 1

            #decrease lambda
            if(search_lambda_dec == 1):
                if(torch.norm(z,1) == 0):
                    return 0
                else:
                    return lambda1

            alpha_y = alpha_y * rho
            # cond1 = F(z, loss_type, net, lambda1, original_img,target_class,tar) <= F(y, loss_type, net, lambda1, original_img,target_class,tar) - delta*(torch.norm(z-y,2)*torch.norm(z-y,2))
            # cond2 = F(z, loss_type, net, lambda1, original_img,target_class,tar) <= c - delta*(torch.norm(z-y,2)*torch.norm(z-y,2))
            print("Finding conditions")
            cond1 = G(z, original_img, original_class, net, lambda1) <= G(y, original_img, original_class, net, lambda1) - delta*(torch.norm(z-y,2)*torch.norm(z-y,2))
            cond2 = G(z, original_img, original_class, net, lambda1) <= c - delta*(torch.norm(z-y,2)*torch.norm(z-y,2))
            print("done with inner", count1)
            if(cond1 | cond2):
                break
        
        # if F(z, loss_type, net, lambda1, original_img,target_class,tar) <= c - delta*(torch.norm(z-y,2)*torch.norm(z-y,2)):
        if G(z, original_img, original_class, net, lambda1) <= c - delta*(torch.norm(z-y,2)*torch.norm(z-y,2)):

            x_pre = x
            temp_norm0 = torch.norm(torch.ones(z.shape).cuda()*z.data,0).item()
            if np.abs(temp_norm0 - x0_norm0) > max_update:
                temp_z = torch.abs((torch.ones(z.shape).cuda()*z.data).reshape(1,-1))
                val, idx = temp_z.topk(k=int(x0_norm0+max_update))

                thres = val[0,int(x0_norm0+max_update-1)]
                z = torch.where(torch.abs(z.data) < thres, torch.zeros(1).cuda(), z.data)
                x = z
            else:
                x = z
        else:

            if k > 0:
                s = x - y_pre.data
                
                grad_x = G_grad(x, original_img, original_class, net)
                
                GRAD_EVALUATIONS += 1

                grad_y_pre = G_grad(y_pre, original_img, original_class, net)
                
                GRAD_EVALUATIONS += 1
            
                r = grad_x - grad_y_pre

                if torch.norm(s, 1) < 1e-5:
                    s = torch.ones(1).cuda() * 1e-5

                if torch.norm(r,1) < 1e-10:
                  r = torch.ones(1).cuda()*1e-10
                
                alpha_x = torch.sum(s*r)/torch.sum(r*r)
                alpha_x = alpha_x.item()

            temp_alpha = alpha_x

        
            if temp_alpha < alpha_min:
                temp_alpha = alpha_min

            if temp_alpha > alpha_max:
                temp_alpha = alpha_max
            if np.isnan(temp_alpha):
                temp_alpha = alpha_min
            alpha_x = temp_alpha

            count2 = 0
            while True:
                count2 = count2 + 1

                if count2 > 10:
                    break
                
                grad_x = G_grad(x, original_img, original_class, net)

                
                GRAD_EVALUATIONS += 1

                v = prox_pixel(x=x-alpha_x*grad_x,alpha=alpha_x,lambda1=lambda1,original_img=original_img, max_epsilon=max_epsilon)
                alpha_x = rho * alpha_x
                # cond3 = F(v, loss_type, net, lambda1, original_img,target_class,tar) <= c - delta*(torch.norm(v-x,2)*torch.norm(v-x,2))
                
                cond3 = G(v, original_img, target_class, net, lambda1) <= c - delta*(torch.norm(v-x,2)*torch.norm(v-x,2))


                if cond3:
                    break
                if torch.abs(G(v, original_img,target_class, net, lambda1) - (c - delta*(torch.norm(v-x,2)*torch.norm(v-x,2)) )) < 1e-3:
                  break

            
            # if F(z, loss_type, net, lambda1, original_img,target_class,tar) <= F(v, loss_type, net, lambda1, original_img,target_class,tar):
            if G(z, original_img, target_class, net, lambda1) <= G(v, original_img, target_class, net, lambda1):
    
                x_pre = x
                temp_norm0 = torch.norm(torch.ones(z.shape).cuda() * z.data, 0).item()
                if np.abs(temp_norm0 - x0_norm0) > max_update:
                    temp_z = torch.abs((torch.ones(z.shape).cuda() * z.data).reshape(1, -1))
                    val, idx = temp_z.topk(k=int(x0_norm0 + max_update))

                    thres = val[0, int(x0_norm0 + max_update - 1)]
                    z = torch.where(torch.abs(z.data) < thres, torch.zeros(1).cuda(), z.data)
                    x = z
                else:
                    x = z
            else:
                x_pre = x
                temp_norm0 = torch.norm(torch.ones(v.shape).cuda() * v.data, 0).item()
                if np.abs(temp_norm0 - x0_norm0) > max_update:
                    temp_v = torch.abs((torch.ones(v.shape).cuda() * v.data).reshape(1, -1))
                    val, idx = temp_v.topk(k=int(x0_norm0 + max_update))
                    thres = val[0, int(x0_norm0 + max_update - 1)]
                    v = torch.where(torch.abs(v.data) < thres, torch.zeros(1).cuda(), v.data)
                    x = v
                else:
                    x = v


        thres = max_epsilon
        x = torch.clamp(x.data,-thres,thres)
        temp_img = original_img.data + x.data
        temp_img = torch.clamp(temp_img.data, 0, 1)
        x = temp_img.data - original_img.data

        y_pre = y.data
        t = (np.sqrt(4*t*t+1)+1)/2
        q = eta*q + 1
        # c = (eta*q*c + F(x, loss_type, net, lambda1, original_img,target_class,tar))/q
        c = (eta*q*c + G(x, original_img,target_class, net, lambda1))/q

        k = k + 1

        pre_loss = cur_loss
        
        logist = net(x.data+original_img.data)
        _,target=torch.max(logist,1)
        
        print("x normmm", torch.norm(x), torch.norm(x, 0))
        cur_loss = G_loss(x, original_img, original_class, net)

        #success
        if tar == 1:
          if(target == target_class):
            success = 1
            break
        else:
          if((target != target_class)):
            success = 1
            break
        print("Curr loss: ", cur_loss)
        if ((success == 0) and (k >= maxiter) and (np.abs(pre_loss-cur_loss) < 1e-3) and (counter==1)):
            break

        if((k >= maxiter) and (np.abs(pre_loss-cur_loss) < 1e-3)):
            counter = 1

    return x, success



def search_lambda(loss_type, net, original_img, target_class, original_class, tar, val_c, max_update, maxiter, max_epsilon):
    global GRAD_EVALUATIONS
    lambda1 = 1e-2
    x0 = torch.zeros(original_img.shape).cuda()
    # x0 = torch.randn_like(original_img).cuda() * 0.001
    print("normmm: ", torch.norm(x0))

    k1 = 0
    while True:
        k1 = k1 + 1
        temp = nmAPG(x0=x0, loss_type=loss_type, net=net, eta=0.9, delta=0.3, rho=0.8, original_img=original_img, lambda1=lambda1,
                     search_lambda_inc=1, search_lambda_dec=0, target_class=target_class, original_class=original_class, tar=tar, max_update=max_update, maxiter=maxiter, max_epsilon=max_epsilon)
        if temp == 0:
            lambda1 = lambda1 + 1e-2
        if temp == 1:
            break
        
        print("incr search iteration ", k1, lambda1,temp)


    print("done with increment")
    k2 = 0
    while True:
        k2 = k2 + 1
        temp = nmAPG(x0=x0, loss_type=loss_type, net=net, eta=0.9, delta=0.3, rho=0.8, original_img=original_img, lambda1=lambda1,
                     search_lambda_inc=0, search_lambda_dec=1, target_class=target_class, original_class=original_class, tar=tar,max_update=max_update, maxiter=maxiter, max_epsilon=max_epsilon)
        if temp == 0:
            lambda1 = lambda1 * 0.99
        else:
            break
        
        print("decr search iteration ", k2, lambda1)


    lambda1 = lambda1 * val_c
    print('attack lambda = ', lambda1)

    return lambda1

def homotopy(loss_type, net, original_img, target_class, original_class, tar, max_epsilon, dec_factor, val_c, val_w1, val_w2, max_update, maxiter, val_gamma):
    global GRAD_EVALUATIONS
    # lambda1 = search_lambda(loss_type, net, original_img, target_class, original_class,tar, val_c, max_update, maxiter, max_epsilon=max_epsilon)
    lambda1 = 0.1
    x = torch.zeros(original_img.shape).cuda()
    # x = torch.randn_like(original_img).cuda() * 0.001
    print("normmm: ", torch.norm(x))

    pre_norm0 = 0
    cur_norm0 = 0

    max_norm0 = torch.norm(torch.ones(x.shape).cuda(),0).item()
    outer_iter = 0
    val_max_update = max_update
    while True:
        outer_iter = outer_iter + 1
        print("Iteration:", outer_iter)

        x, success = nmAPG(x0=x, loss_type=loss_type, net=net, eta=0.9, delta=0.3, rho=0.8, original_img=original_img, lambda1=lambda1,
                           search_lambda_inc=0, search_lambda_dec=0, target_class=target_class, original_class=original_class, tar=tar, max_update=max_update, maxiter=maxiter, max_epsilon=max_epsilon)
        max_update = val_max_update
        pre_norm0 = cur_norm0
        cur_norm0 = torch.norm(torch.ones(x.shape).cuda()*x.data,0).item()
        cur_norm1 = torch.norm(torch.ones(x.shape).cuda() * x.data, 1).item()

        #attack fail
        if(cur_norm0 > max_norm0*0.95 and outer_iter*max_update > max_norm0*0.95):
            break

        iters = 0
        if (cur_norm1 <= cur_norm0 * max_epsilon * val_gamma):
            max_update = 1
            iters = 50 #200
            if cur_norm0 >= 100:
                iters = 100 #400
            if cur_norm0 >= 200:
                iters = 150 #600
            if cur_norm0 >= 300:
                iters = 200 #800
            if cur_norm0 >= 400:
                iters = 250 #1000
            if cur_norm0 >= 500:
                iters = 300 #1200

        if success == 0:
            print("After attack needed")
            # x = after_attack(x, net, original_img, original_class, target_class, post=0, loss_type=loss_type, tar=tar, iters=iters, val_w1=val_w1, val_w2=val_w2, max_epsilon=max_epsilon)
            lambda1 = dec_factor * lambda1
        else:
            break

        logi = net(x.data+original_img.data)
        _,cur_class=torch.max(logi,1)
        if tar == 1:
            if((cur_class == target_class)):
                break
        else:
            if((cur_class != target_class)):
                break

    # x = after_attack(x, net, original_img, original_class, target_class, post=1, loss_type=loss_type,tar=tar, iters=iters, val_w1=val_w1, val_w2=val_w2, max_epsilon=max_epsilon)

    return x





def main():
    args = BaseParser().parse()

    #model loading
    # net = inception_v3.inception_v3(pretrained=False)
    # net.load_state_dict(torch.load('./pretrain/inception_v3_google-1a9a5a14.pth'))
    # net.eval()
    # net.cuda()
    
    net = resnet.ResNet18()
    net = net.to('cuda')
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    checkpoint = torch.load('./pretrain/resnet.pth')
    net.load_state_dict(checkpoint['net'])
    net.eval()
    # net.cuda()

    test_dataset = BaseDataset(
        args.imgdir,
        transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batchSize, shuffle=False)

    root = os.path.join('./test/', args.name)

    counter = 0
    sum_l1 = 0
    sum_l2 = 0
    sum_l_inf = 0
    sum_l0 = 0
    total_start_time = time.time()
    num_success = 0


    for idx, data in enumerate(test_loader):

        input = data['img']
        input = input.cuda()
        ori_image = Variable(input,requires_grad = False)


        image_names = data['name']

        # if int(idx) == 0:
        #     continue

        print(idx)
        print(image_names)
        if ori_image.shape[0] != 1 or  ori_image.shape[1] != 3  or ori_image.shape[2] != 32 or ori_image.shape[3] != 32:
            continue


        logist = net(ori_image)
        _,target=torch.max(logist,1)
        print('original class = ', target)

        original_class = target.item()


        '''
        if(original_class != labels[idx]):
            continue
        '''

        print('target = ', args.target_class)
        if original_class == args.target_class:
            continue

        counter = counter + 1

        iter_start_time = time.time()

        if args.target_class == -1:
            adv = homotopy(loss_type='ce', net=net, original_img=ori_image, target_class=original_class, original_class=original_class, tar=0, max_epsilon=args.max_epsilon,
                       dec_factor=args.dec_factor, val_c=args.val_c, val_w1=args.val_w1, val_w2=args.val_w2, max_update=args.max_update, maxiter=args.maxiter, val_gamma=args.val_gamma)
        else:
            adv = homotopy(loss_type='ce', net=net, original_img=ori_image, target_class=args.target_class, original_class=original_class, tar=1, max_epsilon=args.max_epsilon,
                       dec_factor=args.dec_factor, val_c=args.val_c, val_w1=args.val_w1, val_w2=args.val_w2, max_update=args.max_update, maxiter=args.maxiter, val_gamma=args.val_gamma)

        end_time = time.time()

        print('total time = ', end_time-iter_start_time)

        adv = adv.data+ori_image.data

        logist = net(adv.data)
        _,target=torch.max(logist,1)
        print('after attack class = ',target)

        noise = adv.data - ori_image.data


        cur_l0 = int(torch.norm(noise.data, 0).item())
        cur_l1 = float(torch.norm(noise.data, 1).item())
        cur_l2 = float(torch.norm(noise.data, 2).item())
        cur_l_inf = float(torch.norm(noise.data, p=float("inf")).item())

        # if cur_l0 > 0 and args.target_class == target.item():
        if cur_l0 > 0:
            num_success = num_success + 1
            sum_l1 = sum_l1 + cur_l1
            sum_l2 = sum_l2 + cur_l2
            sum_l_inf = sum_l_inf + cur_l_inf
            sum_l0 = sum_l0 + cur_l0


        print('L0 = ', cur_l0)
        print('L1 = ', cur_l1)
        print('L2 = ', cur_l2)
        print('L-inf = ', cur_l_inf)
        print("Total gradient calculations: ", GRAD_EVALUATIONS)

        # exit(0)

        if not os.path.exists(root):
            os.makedirs(root)
        if not os.path.exists(os.path.join(root,'benign')):
            os.makedirs(os.path.join(root,'benign'))
        if not os.path.exists(os.path.join(root,'adv')):
            os.makedirs(os.path.join(root,'adv'))


        

        for i in range(input.size(0)):
            real_img = ToPILImage()(ori_image[i])

            adv_path = os.path.join(root, 'adv', image_names[i] + '_' + 'adv' + '.png')

            adv_img = transforms.ToPILImage()(adv.cpu()[0]).convert('RGB')
            adv_img.save(adv_path)

            real_path = os.path.join(root,'benign', image_names[i] +'_' + 'ori' + '.png')
            real_img.save(real_path)

    print()
    print('counter = ', counter)
    print('num_success = ', num_success)
    print('sum_l0 = ', sum_l0)
    print('sum_l1 = ', sum_l1)
    print('sum_l2 = ', sum_l2)
    print('sum_l_inf = ', sum_l_inf)
    print('total time = ', time.time()-total_start_time)



def CWLoss(logits, target, kappa=0, tar = True):
    target = torch.ones(logits.size(0)).type(torch.cuda.FloatTensor).mul(target.float())
    target_one_hot = Variable(torch.eye(10).type(torch.cuda.FloatTensor)[target.long()].cuda())
    
    real = torch.sum(target_one_hot*logits, 1)
    other = torch.max((1-target_one_hot)*logits - (target_one_hot*10000), 1)[0]
    kappa = torch.zeros_like(other).fill_(kappa)
    
    if tar:
        return torch.sum(torch.max(other-real, kappa))
    else :
        return torch.sum(torch.max(real-other, kappa))


if __name__ == '__main__':
    
    seed = 4204
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    
    main()
