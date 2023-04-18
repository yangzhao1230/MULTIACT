import torch
import torch.nn as nn
import copy
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d

from data.datautils.babel_label import label as BABEL_label
from data.datautils.babel_label import label_over_twenty
from data.datautils.transforms import *

from models.modules.transformer import CVAETransformerEncoder, CVAETransformerDecoder
from models.modules.postprocess import Postprocess
from models.modules.priornet import PriorNet

from utils.human_models import SMPLH
from utils.loss import ParamLoss, CoordLoss, KLLoss, AccelLoss

BABEL_label_rev = {v:k for k, v in BABEL_label.items()}
babel_over_twenty_numbers = sorted([BABEL_label[lab] for lab in label_over_twenty])
babel_from_zero = {numlab: i for i, numlab in enumerate(babel_over_twenty_numbers)}


class CVAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.param_loss = ParamLoss(cfg.loss_type, cfg.loss_dim)
        self.coord_loss = CoordLoss(cfg.loss_type, cfg.loss_dim)
        self.kl_loss = KLLoss(cfg)
        self.accel_loss = AccelLoss()
        self.spec = cfg.Transformer_spec
        self.smpl_model = SMPLH(cfg)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_random_(p)

        batch_index = torch.tensor([[i]*cfg.max_input_len for i in range(60)]).reshape(-1)
        self.register_buffer("batch_index", batch_index, persistent=False)

        frame_index = torch.tensor([[j for j in range(cfg.max_input_len)] for i in range(60)]).reshape(-1)
        self.register_buffer("frame_index", frame_index, persistent=False)
        
        self.smplh_neutral_layer = copy.deepcopy(self.smpl_model.layer).cuda().requires_grad_(False)
        smplh_joint_regressor = self.smpl_model.joint_regressor.clone().detach().cuda()
        self.register_buffer("smplh_joint_regressor", smplh_joint_regressor, persistent=False)
        
        self.priornet = PriorNet(cfg, self.spec["embed_dim"])

        if cfg.encoder == "Transformer":
            self.encoder = CVAETransformerEncoder(cfg, self.spec["embed_dim"])
        else:
            raise NotImplementedError("unknown encoder")

        if cfg.decoder == "Transformer":
            self.decoder = CVAETransformerDecoder(cfg, self.spec["embed_dim"])
        else:
            raise NotImplementedError("unknown decoder")

        if cfg.postprocessor != "none":
            self.postprocessor = Postprocess(cfg)



    def forward(self, inputs, mode):
        # information from batch
        batch_size, input_duration = inputs["smpl_param"].shape[:2]
        # print(f'input duration:{input_duration}')
        smpl_trans = inputs["smpl_trans"]
        smpl_param = inputs["smpl_param"]
        label_mask = inputs["label_mask"]
        labels = inputs["labels"]
        frame_length = inputs["frame_length"]

        S1_end_mask = inputs["S1_end_mask"]
        valid_mask = inputs["valid_mask"]
        subaction_mask = inputs["subaction_mask"]
        output_mask = inputs["output_mask"]

        if mode != "gen":
            subaction_mean_mask = inputs["subaction_mean_mask"]

        if self.cfg.input_rotation_format == "6dim":
            input_param = matrix_to_rotation_6d(axis_angle_to_matrix(smpl_param.reshape(batch_size, input_duration, 52, 3))).reshape(batch_size, input_duration, 312)
        elif self.cfg.input_rotation_format == "axis":
            input_param = smpl_param
        
        input_trans = smpl_trans

        # pre-coding
        encoder_input = torch.cat((input_param, input_trans), dim=2).to(torch.float32)

        S1_end = encoder_input[S1_end_mask == 1].reshape(batch_size, self.cfg.S1_end_len, -1)

        # prior net: prior_mu, prior_logvar will be used in kl loss
        prior_mean, prior_logvar = self.priornet(S1_end, labels[:, 1:])

        # encoder
        if mode != "gen":
            posterior_mean, posterior_logvar = self.encoder(encoder_input, label_mask, valid_mask, subaction_mean_mask)

        if mode == "train":
            z = self.reparameterize(posterior_mean, posterior_logvar)
        elif mode == "gen":
            z = self.reparameterize(prior_mean, prior_logvar)

        if self.cfg.layered_pos_enc:
            transition_len = frame_length[:, 1]
        else:
            transition_len = None
        
        # decoder
        param_decoder_out, trans_decoder_out = self.decoder(z, (batch_size, input_duration, self.spec["embed_dim"]), output_mask, transition_len)
        
        if self.cfg.postprocessor != "none":
            # fill last frame pose into invalid timeframes
            if self.cfg.model_fill_last_frame:
                last_frame_index = torch.sum(frame_length, dim=1) - 1
                last_frame_param = param_decoder_out[torch.arange(0, batch_size), last_frame_index]
                last_frame_trans = trans_decoder_out[torch.arange(0, batch_size), last_frame_index]
                
                output_6d_param = last_frame_param.unsqueeze(1).expand((-1, input_duration, -1)).clone()
                output_trans = last_frame_trans.unsqueeze(1).expand((-1, input_duration, -1)).clone()

            else:
                output_6d_param = torch.zeros_like(smpl_param, dtype=torch.float32, device=smpl_param.device)
                output_trans = torch.zeros_like(smpl_trans, dtype=torch.float32, device=smpl_trans.device)
            # S1 mask 사용 대입
            s1_mask = subaction_mask == 0

            # 3_dim S1 -> 6_dim S1
            s1_smpl_6d = smpl_param[s1_mask].reshape(-1, 52, 3)
            s1_smpl_6d = axis_angle_to_matrix(s1_smpl_6d)
            s1_smpl_6d = matrix_to_rotation_6d(s1_smpl_6d).reshape(-1, 52*6)

            output_6d_param[s1_mask] = s1_smpl_6d
            output_trans[s1_mask] = smpl_trans[s1_mask]

            # decoder output에 output filter 씌워서 output 내용 추출
            # output param에 trs2 mask 씌워서 output 덮어씌울 위치 특정
            output_trs2_mask = output_mask > 0
            trs2_mask = (subaction_mask > 0)
            
            output_6d_param[trs2_mask] = param_decoder_out[output_trs2_mask]
            output_trans[trs2_mask] = trans_decoder_out[output_trs2_mask]

            # postprocessor
            output_6d_param, output_trans = self.postprocessor(output_6d_param, output_trans)
        
        else:
            # just append
            output_6d_param = torch.zeros((batch_size, input_duration, 312), dtype=torch.float32, device=smpl_param.device)
            output_trans = torch.zeros_like(smpl_trans, dtype=torch.float32, device=smpl_trans.device)
            output_trs2_mask = output_mask > 0
            trs2_mask = (subaction_mask > 0)
            
            output_6d_param[trs2_mask] = param_decoder_out[output_trs2_mask]
            output_trans[trs2_mask] = trans_decoder_out[output_trs2_mask]

        output_6d_param = output_6d_param.reshape(-1, 6)
        output_param = rot6d_to_axis_angle(output_6d_param)
        output_param = output_param.reshape(batch_size, self.cfg.max_input_len, -1)

        if self.cfg.postprocessor == "none":
            output_param[subaction_mask==0] = smpl_param[subaction_mask==0]
            output_trans[subaction_mask==0] = smpl_trans[subaction_mask==0]
        
        # gt body model parameters
        gt_body_params = {
            "global_orient": smpl_param[:, :, :3],
            "body_pose": smpl_param[:, :, 3:66],
            "hand_pose": smpl_param[:, :, 66:156],
            "transl": smpl_trans
        }

        # output body model parameters
        out_body_params = {
            "global_orient": output_param[:, :, :3],
            "body_pose": output_param[:, :, 3:66],
            "hand_pose": output_param[:, :, 66:156],
            "transl": output_trans
        }

        gt_smpl_mesh_cam = self.get_batch_smpl_mesh_cam(gt_body_params)
        gt_smpl_mesh_cam = gt_smpl_mesh_cam.reshape(batch_size, self.cfg.max_input_len, -1, 3)

        out_smpl_mesh_cam = self.get_batch_smpl_mesh_cam(out_body_params)
        out_smpl_mesh_cam = out_smpl_mesh_cam.reshape(batch_size, self.cfg.max_input_len, -1, 3)

        gt_smpl_joint_cam = torch.matmul(self.smplh_joint_regressor, gt_smpl_mesh_cam)  # batch * self.cfg.input_duration * self.smpl_model.orig_joint_num * 3, use joint regressor
        out_smpl_joint_cam = torch.matmul(self.smplh_joint_regressor, out_smpl_mesh_cam)  # batch * self.cfg.input_duration * self.smpl_model.orig_joint_num * 3, use joint regressor
    

        # generated transition outputs
        trans_out_smpl_joint_cam = torch.zeros_like(out_smpl_joint_cam)
        trans_out_smpl_joint_cam[output_mask==1] = out_smpl_joint_cam[subaction_mask==1]

        # gt transition outputs
        gt_trans_smpl_gt_cam = torch.zeros_like(gt_smpl_joint_cam)
        gt_trans_smpl_gt_cam[output_mask==1] = gt_smpl_joint_cam[subaction_mask==1]


        # wipe out invalid timeframe
        not_valid_mask = subaction_mask == -1
        gt_smpl_mesh_cam[not_valid_mask] = 0
        gt_smpl_joint_cam[not_valid_mask] = 0
        out_smpl_mesh_cam[not_valid_mask] = 0
        out_smpl_joint_cam[not_valid_mask] = 0

        accel_loss = self.accel_loss(out_smpl_joint_cam, valid_mask)

        loss = {}
        out = {}

        if mode == "train":
            loss['rec_pose'] = self.param_loss(
                torch.cat((output_param, output_trans), dim=2).to(torch.float32), 
                torch.cat((smpl_param, smpl_trans), dim=2).to(torch.float32),
                valid=valid_mask.unsqueeze(2).expand(-1, -1, 159)
            )
            loss['rec_mesh'] = self.coord_loss(out_smpl_mesh_cam, gt_smpl_mesh_cam) * self.cfg.mesh_weight
            loss['accel_loss'] = accel_loss * self.cfg.accel_weight
            loss['kl_loss'] = self.kl_loss(posterior_mean, posterior_logvar, prior_mean, prior_logvar) * self.cfg.kl_weight
        
        out['gen_smpl_mesh'] = out_smpl_mesh_cam
        out['gen_smpl_joint'] = out_smpl_joint_cam
        out['gt_smpl_mesh'] = gt_smpl_mesh_cam
        out['gt_smpl_joint'] = gt_smpl_joint_cam
        out['gen_smpl_param'] = output_param
        out['gen_smpl_trans'] = output_trans

        out['gt_transition_joint'] = gt_trans_smpl_gt_cam
        out['gen_transition_joint'] = trans_out_smpl_joint_cam

        return loss, out


    def generate_gaussian_prior(self, num_samples):
        mean = torch.zeros((num_samples, self.spec["embed_dim"]))
        logvar = torch.zeros((num_samples, self.spec["embed_dim"]))
        z = self.reparameterize(mean, logvar)
        return z

    def reparameterize(self, mu, logvar):
        s_var = logvar.mul(0.5).exp_()
        eps = s_var.data.new(s_var.size()).normal_()
        return eps.mul(s_var).add_(mu)

    def get_batch_smpl_mesh_cam(self, body_params):
        smpl_out = self.smplh_neutral_layer(
            pose_body=body_params["body_pose"].reshape(-1, 63).to(torch.float32),
            pose_hand=body_params["hand_pose"].reshape(-1, 90).to(torch.float32),
            root_orient=body_params["global_orient"].reshape(-1, 3).to(torch.float32) if body_params["global_orient"] != None else None,
            trans=body_params["transl"].reshape(-1, 3).to(torch.float32)
        )
        return smpl_out.v

class MDM(nn.Module):
    def __init__(self, cfg, **kargs):
        # nfeats, action_profiles, njoints=159,
        #          latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
        #          activation="gelu", 
        #          **kargs):
        super().__init__()

        # self.cond_mask_prob = cfg.cond_mask_prob

        self.njoints = cfg.mdm_spec['njoints']
        self.nfeats = cfg.mdm_spec['nfeats']
        self.num_actions = len(cfg.action_profiles)
        self.latent_dim = cfg.mdm_spec['latent_dim']

        self.ff_size = cfg.mdm_spec['ff_size']
        self.num_layers = cfg.mdm_spec['num_layers']
        self.num_heads = cfg.mdm_spec['num_heads']
        self.dropout = cfg.mdm_spec['dropout']

        self.activation = cfg.mdm_spec['activation']

        self.input_feats = self.njoints * self.nfeats

        self.cond_mode = cfg.mdm_spec['cond_mode']
        self.cond_mask_prob = cfg.mdm_spec['cond_mask_prob']

        self.input_process = InputProcess(self.input_feats, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        # print("TRANS_ENC init")
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=self.num_layers)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        if self.cond_mode != 'no_cond':
            if 'action' in self.cond_mode:
                self.embed_action = EmbedAction(self.num_actions, self.latent_dim)
                print('EMBED ACTION')

        self.output_process = OutputProcess(self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)
        self.smpl_model = SMPLH(cfg)
        self.smplh_neutral_layer = copy.deepcopy(self.smpl_model.layer).cuda().requires_grad_(False)
        smplh_joint_regressor = self.smpl_model.joint_regressor.clone().detach().cuda()
        self.register_buffer("smplh_joint_regressor", smplh_joint_regressor, persistent=False)

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats, nframes = x.shape # [bs, 159, 1, 120]
        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        force_mask = y.get('uncond', False)
        if 'text' in self.cond_mode:
            enc_text = self.encode_text(y['text'])
            emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))
        if 'action' in self.cond_mode:
            action_emb = self.embed_action(y['action'])
            emb += self.mask_cond(action_emb, force_mask=force_mask)

        x = self.input_process(x)

        mask = y['mask']
        token_mask = torch.ones((bs, 1), dtype=bool, device=x.device)
        aug_mask = torch.cat((token_mask, mask), 1)
        xseq = torch.cat((emb, x), axis=0)
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        output = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)[1:]

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output
    
    def get_batch_smpl_mesh_cam(self, body_params):
        smpl_out = self.smplh_neutral_layer(
            pose_body=body_params["body_pose"].reshape(-1, 63).to(torch.float32),
            pose_hand=body_params["hand_pose"].reshape(-1, 90).to(torch.float32),
            root_orient=body_params["global_orient"].reshape(-1, 3).to(torch.float32) if body_params["global_orient"] != None else None,
            trans=body_params["transl"].reshape(-1, 3).to(torch.float32)
        )
        return smpl_out.v
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        # self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        # if self.data_rep == 'rot_vel':
        #     self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats) # [120 bs 159]

        # if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x


class OutputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        output = self.poseFinal(output)  # [seqlen, bs, 150]
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        # idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[input]
        return output
    
def get_model(cfg):
    if cfg.model == "CVAE":
        return CVAE(cfg)
    elif cfg.model == 'MDM':
        return MDM(cfg)
    else:
        raise NotImplementedError("unknown model")
    
def lengths_to_mask(lengths, device: torch.device):
    lengths = torch.tensor(lengths, device=device)
    max_len = max(lengths)
    mask = torch.arange(max_len, device=device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    
