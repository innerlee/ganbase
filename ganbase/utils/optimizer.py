import torch.optim as optim


def get_optimizer(params, optimizer,**kwargs):
    """
    get an optimizer, accepted args are,

    for adam, adamax
        * params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0

    for rmsprop
        * params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False

    for sgd
        * params, lr=0.1, momentum=0, dampening=0, weight_decay=0, nesterov=False
    """
    if optimizer == 'adam':
        args = {
            'lr': 1e-3,
            'beta1': 0.9,
            'beta2': 0.999,
            'eps': 1e-8,
            'weight_decay': 0
        }
        a = {key: kwargs[key] if key in kwargs else args[key] for key in args}
        return optim.Adam(
            params,
            lr=a['lr'],
            betas=(a['beta1'], a['beta2']),
            eps=a['eps'],
            weight_decay=a['weight_decay'])
    if optimizer == 'adamax':
        args = {
            'lr': 1e-3,
            'beta1': 0.9,
            'beta2': 0.999,
            'eps': 1e-8,
            'weight_decay': 0
        }
        a = {key: kwargs[key] if key in kwargs else args[key] for key in args}
        return optim.Adamax(
            params,
            lr=a['lr'],
            betas=(a['beta1'], a['beta2']),
            eps=a['eps'],
            weight_decay=a['weight_decay'])
    if optimizer == 'rmsprop':
        args = {
            'lr': 1e-2,
            'alpha': 0.99,
            'eps': 1e-8,
            'weight_decay': 0,
            'momentum': 0,
            'centered': False
        }
        a = {key: kwargs[key] if key in kwargs else args[key] for key in args}
        return optim.RMSprop(params, **a)
    if optimizer == 'sgd':
        args = {
            'lr': 0.1,
            'momentum': 0,
            'dampening': 0,
            'weight_decay': 0,
            'nesterov': False
        }
        a = {key: kwargs[key] if key in kwargs else args[key] for key in args}
        return optim.SGD(params, **a)
    else:
        raise ValueError(f'optimizer `{optimizer}` not supported')
