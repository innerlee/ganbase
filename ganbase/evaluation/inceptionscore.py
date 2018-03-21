import math
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.models.inception import inception_v3


def inception_score(imgs, batch_size=10, splits=10):
    """Computes the inception score of the generated images imgs
    imgs -- List of tensors in (N,C,H,W)
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """

    N = len(imgs)
    assert batch_size > 0
    assert N > batch_size

    # load inception model
    dtype = torch.cuda.FloatTensor

    inception_model = inception_v3(
        pretrained=True, transform_input=False).type(dtype)
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
        input_batch = input_imgs[(i * batch_size):min((i + 1) * batch_size, N)]
        input_batch = Variable(torch.cat(input_batch, 0).type(dtype))
        pred[(i * batch_size):min((i + 1) * batch_size, N)] = F.softmax(
            inception_model(input_batch), dim=1).data

    print("Compute Inception Score")
    scores = []
    for j in range(splits):
        part = pred[j * (N // splits):(j + 1) * (N // splits), :]
        py = torch.mean(part, 0)
        temp = torch.zeros(N // splits)

        for k in range(part.size()[0]):
            py_x = part[k, :]
            kl = torch.sum(py_x * torch.log(py_x / py))
            temp[k] = kl

        scores.append(torch.exp(torch.FloatTensor([torch.mean(temp)])))

    scores = torch.cat(scores, 0)
    return torch.mean(scores), torch.std(scores)
