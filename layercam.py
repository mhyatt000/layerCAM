""" Part of code borrows from https://github.com/1Konny/gradcam_plus_plus-pytorch """
import time

import numpy as np
import torch
import torch.nn.functional as F

import utils
from utils.layers import (find_alexnet_layer, find_densenet_layer,
                          find_googlenet_layer, find_layer,
                          find_mobilenet_layer, find_resnet_layer,
                          find_shufflenet_layer, find_squeezenet_layer,
                          find_vgg_layer)

# from uutils import deco


class LayerCAM:
    def __init__(self, model_dict):
        name = model_dict["name"].lower()
        layers = model_dict["layers"]
        # layers = [f"features_{l}" for l in layers]

        self.ifcuda = torch.cuda.is_available()
        self.model = model_dict["model"].cuda() if self.ifcuda else model_dict["model"]
        self.model.eval()

        self.gradients, self.activations = {}, {}

        def backward_hook(layer):
            def fn(module, grad_input, grad_output):
                self.gradients[layer] = grad_output[0].cuda() if self.ifcuda else grad_output[0]
            return fn

        def forward_hook(layer):
            def fn(module, input, output):
                self.activations[layer] = output.cuda() if self.ifcuda else output
            return fn

        find = {
            "vgg": find_vgg_layer,
            "resnet": find_resnet_layer,
            "densenet": find_densenet_layer,
            "alexnet": find_alexnet_layer,
            "squeezenet": find_squeezenet_layer,
            "googlenet": find_googlenet_layer,
            "shufflenet": find_shufflenet_layer,
            "mobilenet": find_mobilenet_layer,
        }

        target = False
        for k, func in find.items():
            if k in name:
                self.layers = {layer:func(self.model, layer) for layer in layers}
                target = True

        if not target:
            self.layers = {layer:find_layer(self.model, layer) for layer in layers}

        for name,layer in self.layers.items():
            layer.register_forward_hook(forward_hook(name))
            layer.register_backward_hook(backward_hook(name))

        self.normcams, self.rel = [],[]
        self.thresholds = [0.05, 0.1, 0.2, 0.3, 0.5]

    def forward(self, input, class_idx=None):
        """builds saliency map"""

        b, c, h, w = input.size()

        start = time.perf_counter()

        # predication on raw input
        logit = self.model(input)
        
        print(logit.shape)
        # quit()

        stop = time.perf_counter()
        print(f"Inferenced in {round(stop-start,2)}")

        if class_idx is None:
            predicted_class = logit.max(1)[-1]
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            predicted_class = torch.LongTensor([class_idx])
            score = logit[:, class_idx].squeeze()

        # logit = F.softmax(logit)

        if self.ifcuda:
            predicted_class = predicted_class.cuda()
            score = score.cuda()
            logit = logit.cuda()

        one_hot_output = torch.FloatTensor(1, logit.size()[-1]).zero_()
        one_hot_output[0][predicted_class] = 1

        if self.ifcuda:
            one_hot_output = one_hot_output.cuda(non_blocking=True)

        # Zero grads
        self.model.zero_grad()

        '''TODO
        look at logits?
        question is where did the model look not what class is it
        '''


        # Backward pass with specified target
        # logit.backward(gradient=one_hot_output, retain_graph=True)
        logit.backward(gradient=logit, retain_graph=True)

        for k, v  in self.layers.items():
            print(k)
            # extract activations from hook
            activations = self.activations[k].clone().detach()

            # extract gradients from hook
            gradients = self.gradients[k].clone().detach()
            b, k, u, v = activations.size()

            with torch.no_grad():

                activation_maps = activations * F.relu(gradients)

                # sum across all channels??
                cam = torch.sum(activation_maps, dim=1).unsqueeze(0)
                cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)

                # normalize
                cam_min, cam_max = cam.min(), cam.max()
                normcam = (cam - cam_min).div(cam_max - cam_min + 1e-8).data

                utils.image.basic_visualize( input.cpu().detach(), normcam)
                self.normcams.append(normcam)

        return self.normcams

    def __call__(self, input, class_idx=None):

        start = time.perf_counter()
        cam = self.forward(input, class_idx)
        stop = time.perf_counter()
        print(f"Finished in {round(stop-start,2)}")
        return cam

    def relevance(self,img):
        """computes relevance maps on an image"""

        topath = lambda i: f"./{args.output}/relevance_{'0' if i<9 else ''}{i}.png"

        _ = None if self.normcams else self(img)
        thresh = np.geomspace(0.03, 0.4, num=len(self.normcams))
        for a,cam in zip(thresh,self.normcams):
            print(a)
            r = torch.where(cam.type(torch.FloatTensor).cpu() > a, 1, 0)
            r = torch.cat((r, r, r)).permute(1, 0, 2, 3).float()

            utils.image.basic_visualize(img.cpu().detach(), r)
            self.rel.append(r)

        return self.rel

