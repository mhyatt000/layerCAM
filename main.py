from argparse import ArgumentParser as AP

import torch
import torch.nn.functional as F
import torchvision.models as models
from tqdm import tqdm

from layercam import LayerCAM
import utils


def get_args():
    ap = AP(description="The Pytorch code of LayerCAM")
    ap.add_argument(
        "-i",
        "--img_path",
        type=str,
        default="images/ILSVRC2012_val_00000476.JPEG",
        help="Path of test image",
    )
    ap.add_argument(
        "-l", "--layer_id", type=list, default=[4, 9, 16, 23, 30], help="The cam generation layer"
    )
    ap.add_argument("-o", "--output", type=str, default='.')

    return ap.parse_args()


def main():

    args = get_args()

    img = utils.image.apply_transforms(utils.image.load_image(args.img_path))
    if torch.cuda.is_available():
        img = img.cuda()

    vgg = models.vgg16(pretrained=True).eval()
    resnet = models.resnet152(pretrained=True).eval()
    vgg = resnet
    print(resnet)
 
    # args.layer_id = [i for i in range(len([m for m in vgg.modules()]))]

    maps = []
    thresholds = [0.05, 0.1, 0.2, 0.3, 0.5]
    layers = [f'features_{id}' for id in args.layer_id]

    for layer, a in tqdm(zip(args.layer_id, thresholds), total=len(args.layer_id)):

        vgg_model_dict = dict(name="vgg16", model=vgg, layers=[*layers])
        layercam = LayerCAM(vgg_model_dict)
        # predicted_class = vgg(img).max(1)[-1]

        cam = layercam(img)
        rel = layercam.relevance(img)
        quit()

        topath = lambda i: f"./{args.output}/stage_{'0' if i<9 else ''}{i}.png"
        utils.image.basic_visualize(
            img.cpu().detach(), cam.type(torch.FloatTensor).cpu(), path=topath(id)
        )

        topath = lambda i: f"./{args.output}/relevance_{'0' if i<9 else ''}{i}.png"
        r = torch.where(cam.type(torch.FloatTensor).cpu() > a, 1, 0)
        r = torch.cat((r, r, r)).permute(1, 0, 2, 3).float()
        utils.image.basic_visualize(img.cpu().detach(), r, path=topath(id))

        maps.append(cam)

    m = torch.sum(torch.cat(maps), dim=0)
    utils.image.basic_visualize(
        img.cpu().detach(),
        m.type(torch.FloatTensor).cpu(),
        path=f"./{args.output}/stage_total.png",
    )


if __name__ == "__main__":
    main()
