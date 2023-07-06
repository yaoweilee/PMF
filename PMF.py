import torch
import argparse
import torch.nn as nn
from transformers.models.vit.modeling_vit import ViTModel
from transformers.models.bert.modeling_bert import BertModel

parser = argparse.ArgumentParser(description='PMF architecture')
parser.add_argument("--bert_sz", type=str, default="base", choices=["base", "large"])
parser.add_argument("--vit_sz", type=str, default="base", choices=["base", "large"])
parser.add_argument('--n_fusion_layers', default=2, type=int, help='number of multimodal fusion layers')
parser.add_argument('--n_qp', default=4, type=int, help='length of query prompt vectors')
parser.add_argument('--n_qcp', default=4, type=int, help='length of query context prompt vectors')
parser.add_argument('--n_fcp', default=4, type=int, help='length of fusion context prompt vectors')
parser.add_argument('--n_classes', default=101, type=int, help='number of classes')
parser.add_argument('--mlp_hidden_sz', default=-1, type=int, help='hidden size of non-linear transformation')
args = parser.parse_args()

class MLP_adapter(nn.Module):
    # Non-Linear Transformation in the paper, acting as the translator between modalities.
    def __init__(self, in_dim:int, hidden_dim:int, out_dim:int):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class PMF(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bert_encoder = BertModel.from_pretrained('bert-{}-uncased'.format(args.bert_sz))
        self.bert_encoder.heads = nn.Linear(self.bert_encoder.config.hidden_size, args.n_classes)
        self.vit_encoder = ViTModel.from_pretrained('google/vit-{}-patch16-224-in21k'.format(args.vit_sz))
        self.vit_encoder.heads = nn.Linear(self.vit_encoder.config.hidden_size, args.n_classes)
        
        self.args = self.check_args(args)

        # v2t: vision-to-text. t2v: text-to-vision
        self.v2t_qp = nn.ParameterList([nn.Parameter(torch.empty(1, args.n_qp, self.vit_encoder.config.hidden_size).normal_(std=0.02)) for _ in range(args.n_fusion_layers)])
        self.t2v_qp = nn.ParameterList([nn.Parameter(torch.empty(1, args.n_qp, self.bert_encoder.config.hidden_size).normal_(std=0.02)) for _ in range(args.n_fusion_layers)])
        
        self.v2t_qcp = nn.ParameterList([nn.Parameter(torch.empty(1, args.n_qcp, self.vit_encoder.config.hidden_size).normal_(std=0.02)) for _ in range(args.n_fusion_layers)])
        self.t2v_qcp = nn.ParameterList([nn.Parameter(torch.empty(1, args.n_qcp, self.bert_encoder.config.hidden_size).normal_(std=0.02)) for _ in range(args.n_fusion_layers)])

        if args.n_fcp > 0:
            self.vision_fcp = nn.ParameterList([nn.Parameter(torch.empty(1, args.n_fcp, self.vit_encoder.config.hidden_size).normal_(std=0.02)) for _ in range(args.n_fusion_layers)])
            self.text_fcp = nn.ParameterList([nn.Parameter(torch.empty(1, args.n_fcp, self.bert_encoder.config.hidden_size).normal_(std=0.02)) for _ in range(args.n_fusion_layers)])

        self.v2t_trans = nn.ModuleList([MLP_adapter(self.vit_encoder.config.hidden_size, args.mlp_hidden_sz, self.bert_encoder.config.hidden_size) for _ in range(args.n_fusion_layers)])
        self.t2v_trans = nn.ModuleList([MLP_adapter(self.bert_encoder.config.hidden_size, args.mlp_hidden_sz, self.vit_encoder.config.hidden_size) for _ in range(args.n_fusion_layers)])

        self.grad_control()

    def forward(self, image, txt_input_ids, txt_token_type_ids, txt_attn_mask):
        n = image.shape[0]
        device = image.device
        assert image.shape[0] == txt_input_ids.shape[0]
        
        # pre_processing before two encoders 
        img_tokens = self.vit_encoder.embeddings(image)
        txt_tokens = self.bert_encoder.embeddings(txt_input_ids, txt_token_type_ids)
        
        ## generate extra txt attn mask
        txt_attn_mask = self.get_extended_txt_attn_mask(txt_attn_mask)
        max_prompt_length = self.args.n_qp + self.args.n_qcp + self.args.n_fcp
        batch_extra_attn_mask = torch.ones(n, max_prompt_length).to(device)
        batch_extra_attn_mask = self.get_extended_txt_attn_mask(batch_extra_attn_mask)
        
        # main forward
        ## unimodal base feature extraction
        for bert_layer_id in range(self.bert_encoder.config.num_hidden_layers-self.args.n_fusion_layers):
            txt_tokens = self.bert_encoder.encoder.layer[bert_layer_id](txt_tokens, txt_attn_mask)[0]
        for vit_layer_id in range(self.vit_encoder.config.num_hidden_layers-self.args.n_fusion_layers):
            img_tokens = self.vit_encoder.encoder.layer[vit_layer_id](img_tokens)[0]
        
        ## multimodal fusion layers
        for fusion_layer_id in range(self.args.n_fusion_layers):
            ### get prompts
            batch_v2t_qp = self.v2t_qp[fusion_layer_id].expand(n, -1, -1).to(device)
            batch_t2v_qp = self.t2v_qp[fusion_layer_id].expand(n, -1, -1).to(device)
            
            batch_v2t_qcp = self.v2t_qcp[fusion_layer_id].expand(n, -1, -1).to(device)
            batch_t2v_qcp = self.t2v_qcp[fusion_layer_id].expand(n, -1, -1).to(device)

            if self.args.n_fcp > 0:
                batch_vision_fcp = self.vision_fcp[fusion_layer_id].expand(n, -1, -1).to(device)
                batch_text_fcp = self.text_fcp[fusion_layer_id].expand(n, -1, -1).to(device)

            ### Query Stage 
            # prepare text attn_mask
            ## slice attn_mask for corresponding text prompts
            layer_t2v_qcp_attn_mask = batch_extra_attn_mask[:,:,:,:self.args.n_qcp]
            layer_t2v_qp_attn_mask = batch_extra_attn_mask[:,:,:,:self.args.n_qp]
            layer_text_fcp_attn_mask = batch_extra_attn_mask[:,:,:,:self.args.n_fcp]
            layer_v2t_qp_attn_mask = batch_extra_attn_mask[:,:,:,:self.args.n_qp]

            ## reform text attn_mask
            query_txt_attn_mask = torch.cat([txt_attn_mask, layer_t2v_qcp_attn_mask, layer_t2v_qp_attn_mask], dim=3)
            fusion_txt_attn_mask = torch.cat([txt_attn_mask, layer_text_fcp_attn_mask, layer_v2t_qp_attn_mask], dim=3)

            # for t2v: get text fusion intermediate hidden-state for ViT
            query_txt_tokens = torch.cat([txt_tokens, batch_t2v_qcp, batch_t2v_qp], dim=1)
            t2v_fusion_intermediate = self.bert_encoder.encoder.layer[bert_layer_id + fusion_layer_id + 1](query_txt_tokens, query_txt_attn_mask)
            t2v_fusion_intermediate = t2v_fusion_intermediate[0][:, -self.args.n_qp:, :]
            t2v_fusion_intermediate = self.t2v_trans[fusion_layer_id](t2v_fusion_intermediate)

            # for v2t: get vision fusion intermediate hidden-state for BERT
            query_img_tokens = torch.cat([img_tokens, batch_v2t_qcp, batch_v2t_qp], dim=1)
            v2t_fusion_intermediate = self.vit_encoder.encoder.layer[vit_layer_id + fusion_layer_id +1](query_img_tokens)
            v2t_fusion_intermediate = v2t_fusion_intermediate[0][:, -self.args.n_qp:, :]
            v2t_fusion_intermediate = self.v2t_trans[fusion_layer_id](v2t_fusion_intermediate)

            ### Fusion Stage
            img_tokens = torch.cat([img_tokens, batch_vision_fcp, t2v_fusion_intermediate], dim=1)
            txt_tokens = torch.cat([txt_tokens, batch_text_fcp, v2t_fusion_intermediate], dim=1)
            
            img_tokens = self.vit_encoder.encoder.layer[vit_layer_id + fusion_layer_id + 1](img_tokens)[0]
            txt_tokens = self.bert_encoder.encoder.layer[bert_layer_id + fusion_layer_id +1](txt_tokens, fusion_txt_attn_mask)[0]

            txt_tokens = txt_tokens[:, :-self.args.n_qp-self.args.n_fcp, :]
            img_tokens = img_tokens[:, :-self.args.n_qp-self.args.n_fcp, :]

        # after main forwards
        txt_tokens = txt_tokens[:, 0]
        img_tokens = self.vit_encoder.layernorm(img_tokens)
        img_tokens = img_tokens[:, 0]

        txt_pred = self.bert_encoder.heads(txt_tokens)
        img_pred = self.vit_encoder.heads(img_tokens)

        return (txt_pred + img_pred)/2

    def get_extended_txt_attn_mask(self, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def check_args(self, args):
        assert args.n_qp > 0
        assert args.n_fusion_layers <= min(self.bert_encoder.config.num_hidden_layers, self.vit_encoder.config.num_hidden_layers)
        if args.mlp_hidden_sz == -1:
            args.mlp_hidden_sz = max(int(self.vit_encoder.config.hidden_size/2), int(self.bert_encoder.config.hidden_size/2))
        return args
    
    def grad_control(self):
        # Does not require grad for parameters other than the introduced trainable modules
        trainable_modules = [self.v2t_qp, self.t2v_qp,
                            self.v2t_qcp, self.t2v_qcp,
                            self.v2t_trans.modules(), self.t2v_trans.modules(),
                            self.bert_encoder.heads.modules(),
                            self.vit_encoder.heads.modules()]
        if hasattr(self, 'fcp'):
            trainable_modules.append(self.vision_fcp)
            trainable_modules.append(self.text_fcp)

        for module in self.modules():
            module.requires_grad_(False)

        for module in trainable_modules:
            for item in module:
                item.requires_grad_(True)  



if __name__ == '__main__':
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    image = torch.randn(1,3,224,224)
    text = tokenizer("it's a beautiful day")
    text_inputs = {}
    text_inputs['input_ids'] = torch.tensor(text['input_ids']).unsqueeze(0)
    text_inputs['token_type_ids'] = torch.tensor(text['token_type_ids']).unsqueeze(0)
    text_inputs['attention_mask'] = torch.tensor(text['attention_mask']).unsqueeze(0)
    model = PMF(args)
    model.eval()
    inputs = [image, text_inputs['input_ids'], text_inputs['token_type_ids'], text_inputs['attention_mask']]
    with torch.no_grad():
        model(*inputs)

    overall_param = sum(p.numel() for p in model.parameters())
    trainable_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    t2v_trans_param = sum(p.numel() for p in model.t2v_trans.parameters() if p.requires_grad)
    v2t_trans_param = sum(p.numel() for p in model.v2t_trans.parameters() if p.requires_grad)
    print(trainable_param/overall_param)
    print((t2v_trans_param + v2t_trans_param) / trainable_param)
