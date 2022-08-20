import torch
import torchvision
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

import foolbox1.foolbox as fb

from datasets import BaseDataset

import resnet

net = resnet.ResNet18()
net = net.to('cuda')
net = torch.nn.DataParallel(net)
cudnn.benchmark = True
checkpoint = torch.load('./pretrain/resnet.pth')
net.load_state_dict(checkpoint['net'])
net.eval()

test_dataset = BaseDataset(
        './ci2',
        transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
    )
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# print('reached here!')

# for idx, data in enumerate(test_loader):

#         input = data['img']
#         input = input.cuda()
#         ori_image = Variable(input,requires_grad = False)


#         image_names = data['name']

#         # if int(idx) == 0:
#         #     continue

#         print(idx)
#         print(image_names)
#         # if ori_image.shape[0] != 1 or  ori_image.shape[1] != 3  or ori_image.shape[2] != 32 or ori_image.shape[3] != 32:
#         #     continue


#         logist = net(ori_image)
#         _,target=torch.max(logist,1)
#         print('original class = ', target)

fmodel = fb.PyTorchModel(net, bounds=(0,1))
images = next(iter(test_loader))['img'].cuda()
labels = torch.tensor([0]).cuda()
print(fb.utils.accuracy(fmodel, images, labels))

init_attack = fb.attacks.SaltAndPepperNoiseAttack(steps=50)
# init_advs = init_attack.run(fmodel, images, labels).cuda()
init_advs, clipped, is_adv = init_attack(fmodel, images, labels, epsilons=0.05)

attack = fb.attacks.PointwiseAttack()
advs = attack.run(fmodel, images, labels, starting_points=init_advs).cuda()
# _, advs, is_adv = attack(fmodel, images, labels, epsilons=0.05)
# print(advs.shape)
print(fb.utils.accuracy(fmodel, advs, labels))

L0_norms = torch.zeros([1, 2]).cuda()
Linf_norms = torch.zeros([1,2]).cuda()
for i in range(advs.shape[0]):
    L0_norms[0,i] = torch.norm(images[i,:,:,:] - advs[i,:,:,:], 0)
    Linf_norms[0,i] = torch.norm(images[i,:,:,:] - advs[i,:,:,:], p=float("inf"))

print(L0_norms)    
print(Linf_norms)
