import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import AutoTokenizer, AutoModel
from torchvision import transforms
import clip
from torchvision.transforms import Compose
from clip.model import ModifiedResNet

from typing import Any, Callable, Optional, Union

import torch

import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import xavier_uniform_
from zero.v1.models.DeocderOnlyTransformer import DecoderOnlyTransformer


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


class ModifiedResNetFeatures(ModifiedResNet):
    '''modified from clip.model.ModifiedResNet'''

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__(layers, output_dim, heads, input_resolution, width)

    def forward(self, x: torch.Tensor):
        x = x.type(self.conv1.weight.dtype)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x0 = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x0)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return {
            "res1": x0,
            "res2": x1,
            "res3": x2,
            "res4": x3,
            "res5": x4,
        }


class tokenizer:
    # 其实这个在预处理的时候用就行了，instruction数据预处理一下
    def __init__(self, model_id='meta-llama/Meta-Llama-3.1-70B-Instruct'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def __call__(self, input):
        return self.tokenizer(input, return_tensors='pt')['input_ids']


class ZeroModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # TODO：确定模型的输入输出，暂定为instruction和image，输出为单帧直接的action
        # image 暂定用没有预训练的resnet18提取的特征用embedding放入transformer
        # instruction 暂定toeknized之后加一层embedding放入transformer
        # TODO：最好不要用embedding层，
        # TODO: 试一试dino2
        # Configurations

        resnet18 = models.resnet18(pretrained=False)
        self.image_encoder_resnet18 = nn.Sequential(*list(resnet18.children())[:-2]).to('cuda')  # remove fc and avgpool

        self.transformer = DecoderOnlyTransformer(config['model'])

        self.transform = transforms.Compose([
            transforms.Resize(224),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # used in RLBench where the light and contrast are not so varied as in real world, so no need to normalize
        ])
        # we assume the embeddings functions like project the input to the transformer's d_model

        self.embedding_text = None  # TODO:
        # 这里也可
        # _, self.encoder_text = get_clip_encoders()

        self.feature_decoder = nn.Sequential(nn.Linear(512, 512),
                                             nn.ReLU(),
                                             nn.Linear(512, 512),
                                             nn.ReLU(),
                                             nn.Linear(512, 512),
                                             nn.ReLU(),
                                             nn.Linear(512, 256),
                                             nn.ReLU(),
                                             nn.Linear(256, 128),
                                             nn.ReLU(),
                                             nn.Linear(128, 64),
                                             nn.ReLU(),
                                             nn.Linear(64, 8),
                                             nn.ReLU()).to('cuda')

        self.action_decoder = nn.Sequential(nn.Linear(1968, 2048),
                                            nn.ReLU(),
                                            nn.Linear(2048, 2048),
                                            nn.ReLU(),
                                            nn.Linear(2048, 2048),
                                            nn.ReLU(),
                                            nn.Linear(2048, 1024),
                                            nn.ReLU(),
                                            nn.Linear(1024, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 128),
                                            nn.ReLU(),
                                            nn.Linear(128, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, 8),).to('cuda')

        # CLIP Related
        '''
        (You can treat this function as a python file to understand its local and global variables)
        Fetch the encoder_image and encoder_text from the CLIP model.
        Note: varible clip is the clip package, varible clip_model is the clip model.
        '''
        self.clip_model, image_transforms = clip.load("RN50")  # TODO: Transform 有问题
        state_dict = self.clip_model.state_dict()
        layers = tuple([len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}")))
                        for b in [1, 2, 3, 4]])
        output_dim = state_dict["text_projection"].shape[1]
        heads = state_dict["visual.layer1.0.conv1.weight"].shape[0] * 32 // 64
        visual_model = ModifiedResNetFeatures(layers, output_dim, heads)
        visual_model.load_state_dict(self.clip_model.visual.state_dict())
        visual_model = visual_model.to('cuda')
        # normalize = clip_transforms.transforms[-1]

        # modify transform

        image_transforms_jian = Compose([
            image_transforms.transforms[0],  # Resize
            image_transforms.transforms[1],  # CenterCrop
            image_transforms.transforms[4],  # Normalize # before normalnized, the image is in range [0,1]
        ])

        for parameters in visual_model.parameters():
            parameters.requires_grad = False

        for parameters in self.clip_model.parameters():
            parameters.requires_grad = False

    def _encoder_text(self, text):
        '''modified from clip.model.CLIP.encode_text
           Only deleted the matrix multiplication with the text projection matrix (because we don't need project, our dim is 512)
        '''
        if type(text) is str or type(text) is list:
            text = clip.tokenize(text).to('cuda')
        elif type(text) is torch.Tensor:
            pass

        x = self.clip_model.token_embedding(text).type(self.clip_model.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        # Jian: Extract the eot embedding, features of end of sentence after transformer(x) is a kind of summary of the sentence please refer the self-attention mechanism's definition
        return x

    def _process_image(self, image):
        image = self.transform(image)
        return image

    def forward(self, instruction, image):
        b, f, c, h, w = image.shape
        image = image.to('cuda')
        image = image.view(b * f, c, h, w)
        image = self._process_image(image)

        image = self.image_encoder_resnet18(image)

        # encode instruction
        instruction = self._encoder_text(instruction)
        text = instruction.view(instruction.size(0), instruction.size(1), -1)

        # encode image
        image = image.view(b, f, image.size(1), image.size(2), image.size(3))
        b, f, d_model, h_f, w_f = image.shape
        image = image.reshape(b, d_model, f * h_f * w_f)

        x = torch.concat((text, image), dim=2).permute(0, 2, 1)
        x = self.transformer(x)  # [batch_size, 5 * 7 * 7 + 1, d_model] [1,246,512]
        x = self.feature_decoder(x)
        x = x.view(b, -1)
        x = self.action_decoder(x)
        return x
