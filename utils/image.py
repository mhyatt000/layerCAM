"""flashtorch.utils

This module provides utility functions for image handling and tensor
transformation.
"""

from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from .imagenet import *


def load_image(image_path):
    """Loads image as a PIL RGB image"""

    return Image.open(image_path).convert("RGB")


def apply_transforms(image, size=224):
    """Transforms a PIL image to torch.Tensor.

    Applies a series of tranformations on PIL image including a conversion
    to a tensor. The returned tensor has a shape of :math:`(N, C, H, W)` and
    is ready to be used as an input to neural networks.

    First the image is resized to 256, then cropped to 224. The `means` and
    `stds` for normalisation are taken from numbers used in ImageNet, as
    currently developing the package for visualizing pre-trained models.

    The plan is to to expand this to handle custom size/mean/std.
    """

    if not isinstance(image, Image.Image):
        image = F.to_pil_image(image)

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ]
    )

    tensor = transform(image).unsqueeze(0)
    tensor.requires_grad = True

    return tensor


def denormalize(tensor):
    """Reverses the normalisation on a tensor.

    Performs a reverse operation on a tensor, so the pixel value range is
    between 0 and 1. Useful for when plotting a tensor into an image.

    Normalisation: (image - mean) / std
    Denormalisation: image * std + mean
    """

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    denorm = tensor.clone()

    for channel, mean, std in zip(denorm[0], means, stds):
        channel.mul_(std).add_(mean)

    return denorm


def standardize_and_clip(tensor, min_value=0.0, max_value=1.0):
    """Standardizes and clips input tensor.

    Standardize the input tensor (mean = 0.0, std = 1.0), ensures std is 0.1
    and clips it to values between min/max (default: 0.0/1.0).
    """

    tensor = tensor.detach().cpu()

    mean = tensor.mean()
    std = tensor.std()
    std += 1e-7 if std == 0 else 0

    standardized = tensor.sub(mean).div(std).mul(0.1)
    clipped = standardized.add(0.5).clamp(min_value, max_value)

    return clipped


def format_for_plotting(tensor):
    """Formats the shape of tensor for plotting.

    Tensors typically have a shape of :math:`(N, C, H, W)` or :math:`(C, H, W)`
    which is not suitable for plotting as images. This function formats an
    input tensor :math:`(H, W, C)` for RGB and :math:`(H, W)` for mono-channel
    data.
    """

    has_batch_dimension = len(tensor.shape) == 4
    out = tensor.clone()

    if has_batch_dimension:
        out = tensor.squeeze(0)

    return out.squeeze(0).detach() if out.shape[0] == 1 else out.permute(1, 2, 0).detach()


def visualize(input_, gradients, path=None, cmap="viridis", alpha=0.7):
    """Method to plot the explanation.  """

    input_ = format_for_plotting(denormalize(input_))
    gradients = format_for_plotting(standardize_and_clip(gradients))

    subplots = [
        ("Input image", [(input_, None, None)]),
        ("Saliency map across RGB channels", [(gradients, None, None)]),
        ("Overlay", [(input_, None, None), (gradients, cmap, alpha)]),
    ]

    num_subplots = len(subplots)

    fig = plt.figure(figsize=(16, 3))

    for i, (title, images) in enumerate(subplots):
        ax = fig.add_subplot(1, num_subplots, i + 1)
        ax.set_axis_off()

        for image, cmap, alpha in images:
            ax.imshow(image, cmap=cmap, alpha=alpha)

        ax.set_title(title)

    _ = plt.show() if not path else plt.savefig(path)


def basic_visualize(input_, gradients, path=None, weight=None, cmap="viridis", alpha=0.7):
    """Method to plot the explanation.  """

    input_ = format_for_plotting(denormalize(input_))
    gradients = format_for_plotting(standardize_and_clip(gradients))

    subplots = [
        ("Saliency map across RGB channels", [(gradients, None, None)]),
        ("Overlay", [(input_, None, None), (gradients, cmap, alpha)]),
    ]

    num_subplots = len(subplots)

    fig = plt.figure(figsize=(4, 4))

    for i, (title, images) in enumerate(subplots):
        ax = fig.add_subplot(1, num_subplots, i + 1)
        ax.set_axis_off()

        for image, cmap, alpha in images:
            ax.imshow(image, cmap=cmap, alpha=alpha)

    _ = plt.show() if not path else plt.savefig(path)

