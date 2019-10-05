from flashtorch.utils import load_image, ImageNetIndex
from flashtorch.utils import apply_transforms, denormalize, format_for_plotting
from flashtorch.saliency import Backprop
from torchvision import models
from networks.models_list import models_list
import torch
import matplotlib.pyplot as plt


def load_model(model, model_path, optimizer=None, resume=False,
               lr=None):
    start_epoch = 0
    checkpoint = torch.load(
        model_path, map_location=lambda storage, loc: storage)
    # print('Loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('network') and not k.startswith('module_list'):
            state_dict[k[8:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                # print('Skip loading parameter {}, required shape{}, '
                #       'loaded shape{}.'.format(
                #           k, model_state_dict[k].shape, state_dict[k].shape))
                state_dict[k] = model_state_dict[k]
        else:
            pass
            # print('Drop parameter {}.'.format(k))
    for k in model_state_dict:
        if not (k in state_dict):
            # print('No param {}.'.format(k))
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            # print('Resumed optimizer with starting learning rate', start_lr)
        else:
            pass
            # print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model


def get_model(model_name):
    if model_name == "alexnet" or model_name == "sqnet1_0" or model_name == "sqnet1_1":
        return models_list[model_name]()
    else:
        return models_list[model_name](pretrained=True)


def show_image(image_address):
    image = load_image(image_address)
    plt.imshow(image)
    plt.title('Original image')
    plt.axis('off')
    return image


def visualise_cnn(model_name, image, class_label, model_path=None, title=None):
    if model_path is None:
        model = get_model(model_name)
    else:
        model = get_model(model_name)
        model = load_model(model, model_path)
    model.eval()
    backprop = Backprop(model)
    # Transform the input image to a tensor
    img = apply_transforms(image)
    # Set a target class from ImageNet task: 24 in case of great gray owl

    imagenet = ImageNetIndex()
    target_class = imagenet[class_label]
    # Ready to roll!
    backprop.visualize(img, target_class, guided=True, title=title)
