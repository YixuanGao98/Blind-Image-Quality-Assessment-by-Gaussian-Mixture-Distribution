import torch
import torch.nn as nn
from torch import nn
from clip_encoder_my import CLIPVisionTower


from sw_models import swin_transformer_v2
class MANIQA(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.model=swin_transformer_v2.SwinTransformerV2(patch_size=4, window_size=24, embed_dim=128, img_size=384, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32),
            is_v2=True, pretrained_window_sizes=[12, 12, 12, 6]
                    ).cuda()

        checkpoint = torch.load(path['swin_model'])

        self.model.load_state_dict(checkpoint['model'], strict=False)


        for param in self.model.parameters():
            param.requires_grad = True#finetune

        self.net=CLIPVisionTower(path['vision_tower_name']).cuda()
        self.net_trans=self.net.vision_tower.base_model.vision_model
        self.embeddings=self.net_trans.embeddings
        self.encoder=self.net_trans.encoder
        self.encoder.requires_grad_(True)
        self.llama_dim_mapper1 = nn.Linear(2816, 1024, bias=False)
        self.llama_dim_mapper2 = nn.Linear(1024, 2816, bias=False)   
        



    def forward(self, x):
        N = x.size()[0]
        x=self.model(x)
        x=x.view(N,-1)

        x_map=(self.llama_dim_mapper1(x))
        x_map=self.encoder(torch.unsqueeze(x_map,1))
        x_map=x_map.last_hidden_state.view(N,-1)#1024
        x_map2=torch.sigmoid(self.llama_dim_mapper2(torch.squeeze(x_map,1)))
        feature=x*x_map2+x
        feature=torch.cat((feature,x),1)
        return feature
