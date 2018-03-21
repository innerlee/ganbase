import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.models.inception import inception_v3
import math

def inception_score(imgs, cuda=True, batch_size=10, splits=10):
    """Computes the inception score of the generated images imgs
    imgs -- List of tensors in (N,C,H,W)
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """

    N = len(imgs)
    assert batch_size > 0
    assert N > batch_size

    # load inception model
    if cuda:
        if not torch.cuda.is_available():
            print("WARNING: You do not have a CUDA device, so use CPU to compute.")
            dtype = torch.FloatTensor
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()

    # process input
    input_imgs = []
    upsample = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)

    for img in imgs:
        if len(img.size()) == 3:
            img = torch.unsqueeze(img, 0)
        if img.size() != torch.Size([1, 3, 299, 299]):
            img = upsample(img)

        input_imgs.append(img.data)


    # get prediction
    pred = torch.zeros((N, 1000))
    n_batches = int(math.ceil(float(N) / float(batch_size)))
    for i in range(n_batches):
        input_batch = input_imgs[(i * batch_size) : min((i + 1) * batch_size, N)]
        input_batch = Variable(torch.cat(input_batch, 0).type(dtype))
        pred[(i * batch_size) : min((i + 1) * batch_size, N)] = F.softmax(inception_model(input_batch), dim=1).data

    print"Compute Inception Score"
    scores = []
    for j in range(splits):
        part = pred[j * (N // splits): (j+1) * (N // splits), :]
        py = torch.mean(part, 0)
        temp = torch.zeros(N // splits)

        for k in range(part.size()[0]):
            py_x = part[k, :]
            kl = torch.sum(py_x * torch.log(py_x/py))
            temp[k] = kl

        scores.append(torch.exp(torch.FloatTensor([torch.mean(temp)])))

    scores = torch.cat(scores, 0)
    return torch.mean(scores), torch.std(scores)


if __name__ == '__main__':

    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)

    import torchvision.datasets as dset
    import torchvision.transforms as transforms

    print "Preparing Dataset..."
    cifar = dset.CIFAR10(root='data/', download=True,
                                 transform=transforms.Compose([
                                     transforms.Scale(32),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                 ])
        )

    imgs = IgnoreLabelDataset(cifar)
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=50000)

    print ("Calculating Inception Score...")

    for i, batch in enumerate(dataloader, 0):
        result = get_inception_score(batch, cuda=True, batch_size=32, splits=10)
        print "Inception Score: {}".format(result[0])
        print "Standard Deviation: {}".format(result[1])
