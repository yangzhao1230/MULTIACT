import torch

def get_cond(length, mask, label):
    micro_cond = {}
    micro_cond['y'] = {}
    micro_cond['y']['length']  = length.cuda()
    micro_cond['y']['mask']  = mask.cuda()
    micro_cond['y']['action']  = label.cuda()
    return micro_cond

def squence_split(cfg, inputs):

    batch_size, input_duration = inputs["smpl_param"].shape[:2]
    smpl_trans = inputs["smpl_trans"]
    smpl_param = inputs["smpl_param"]
    label_mask = inputs["label_mask"]
    labels = inputs["labels"]
    frame_length = inputs["frame_length"] #[bs 3 len]

    S1_end_mask = inputs["S1_end_mask"]
    valid_mask = inputs["valid_mask"]
    subaction_mask = inputs["subaction_mask"]
    output_mask = inputs["output_mask"]
    
    input_param = smpl_param
    input_trans = smpl_trans
    # [bs 120 159]
    encoder_input = torch.cat((input_param, input_trans), dim=2).to(torch.float32).cuda()

    len_s1 = frame_length[:, 0]
    len_tr = frame_length[:, 1]
    len_s2 = frame_length[:, 2]
    
    # [bs, 120]
    canvas = torch.arange(input_duration).unsqueeze(0).expand(batch_size, -1).cuda()
    mask_s1 = canvas < len_s1.unsqueeze(-1)
    mask_tr = canvas < len_tr.unsqueeze(-1)
    mask_s2 = canvas < len_s2.unsqueeze(-1)

    # [bs, 120, 159]
    part_s1 = torch.zeros_like(encoder_input).cuda()
    part_s1[mask_s1] = encoder_input[subaction_mask == 0]
    part_tr = torch.zeros_like(encoder_input).cuda()
    part_tr[mask_tr] = encoder_input[subaction_mask == 1]
    part_s2 = torch.zeros_like(encoder_input).cuda()
    part_s2[mask_s2] = encoder_input[subaction_mask == 2]

    micro_cond_s1 = get_cond(len_s1, mask_s1, labels[:, 0])
    micro_cond_tr = get_cond(len_tr, mask_tr, labels[:, 1])
    micro_cond_s2 = get_cond(len_s2, mask_s2, labels[:, 2])

    part_s1 = part_s1.permute(0, 2, 1).unsqueeze(2)
    part_tr = part_tr.permute(0, 2, 1).unsqueeze(2)
    part_s2 = part_s2.permute(0, 2, 1).unsqueeze(2)
    return part_s1, micro_cond_s1, part_tr, micro_cond_tr, part_s2, micro_cond_s2


# tr_sampel & s2_sample [bs 159 1 120]
def sequence_merge(inputs, tr_sample, tr_mask, s2_sample, s2_mask):

    # sample [bs dim 1 len]
    tr_sample = tr_sample.squeeze().permute(0, 2, 1)
    s2_sample = s2_sample.squeeze().permute(0, 2, 1)

    batch_size, input_duration = inputs["smpl_param"].shape[:2]
    smpl_trans = inputs["smpl_trans"]
    smpl_param = inputs["smpl_param"]
    label_mask = inputs["label_mask"]
    labels = inputs["labels"]
    frame_length = inputs["frame_length"] #[bs 3 len]

    S1_end_mask = inputs["S1_end_mask"]
    valid_mask = inputs["valid_mask"]
    subaction_mask = inputs["subaction_mask"]
    output_mask = inputs["output_mask"]
    
    motion = torch.cat((smpl_param, smpl_trans), dim=2).to(torch.float32).cuda()

    input_param = smpl_param
    input_trans = smpl_trans

    output_tr_mask = subaction_mask == 1
    output_s2_mask = subaction_mask == 2

    motion[output_tr_mask] = tr_sample[tr_mask]
    motion[output_s2_mask] = s2_sample[s2_mask]

    return motion


def get_batch_smpl_mesh_cam(self, body_params):
    smpl_out = self.smplh_neutral_layer(
        pose_body=body_params["body_pose"].reshape(-1, 63).to(torch.float32),
        pose_hand=body_params["hand_pose"].reshape(-1, 90).to(torch.float32),
        root_orient=body_params["global_orient"].reshape(-1, 3).to(torch.float32) if body_params["global_orient"] != None else None,
        trans=body_params["transl"].reshape(-1, 3).to(torch.float32)
    )
    return smpl_out.v

def lengths_to_mask(lengths, max_len, device):
    lengths = torch.tensor(lengths, device=device)
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask

def sync_fn(seq_0, args_0, seq_1, args_1, inter_frames):
    # seq_canavas is the background
    bs, njoints, nfeats, max_frames = seq_0.shape
    # _, _, _, max_frames_1 = seq_1.shape
    # max_frames = max_frames_0 + max_frames_1 - inter_frames
    # len = args_0['length'] + args_1['length'] - inter_frames
    seq_canavas = torch.zeros((bs, njoints, nfeats, max_frames), device=seq_0.device)
    # count record the weight
    count = torch.zeros((bs, max_frames), device=seq_0.device)

    # TODO convert to tensor operation
    for idx in range(bs):
        inter_frame =inter_frames[idx]
        len_0 = args_0['length'][idx]
        len_1 = args_1['length'][idx]
        len = len_0 + len_1 - inter_frame
        seq_canavas[idx,:,:,:len_0] += seq_0[idx,:,:,:len_0]
        seq_canavas[idx,:,:,len_0-inter_frame:len] += seq_1[idx,:,:,:len_1]
        count[idx,:len_0] += 1 
        count[idx, len_0-inter_frame:len] += 1

    count = count.unsqueeze(1).unsqueeze(1)
    seq_comp = seq_canavas / (count + 1e-5)

    # ret_seq_0 = torch.zeros_like(seq_0)
    # ret_seq_1 = torch.zeros_like(seq_1)

    # for idx in range(bs):
    #     len_0 = args_0['length'][idx]
    #     len_1 = args_1['length'][idx]
    #     len = len_0 + len_1 - inter_frames
    #     ret_seq_0[idx,:,:,:len_0] = seq_comp[idx,:,:,:len_0]
    #     ret_seq_1[idx,:,:,:len_1] = seq_comp[idx,:,:,len_0-inter_frames:len]

    return extract_fn(seq_comp, args_0, args_1, inter_frames)

def extract_fn(seq_comp, args_0, args_1, inter_frames):
    bs, njoints, nfeats, _ = seq_comp.shape

    ret_seq_0 = torch.zeros((bs, njoints, nfeats, max(args_0['length'])), device=seq_comp.device)
    ret_seq_1 = torch.zeros((bs, njoints, nfeats, max(args_1['length'])), device=seq_comp.device)

    for idx in range(bs):
        len_0 = args_0['length'][idx]
        len_1 = args_1['length'][idx]
        inter_frame = inter_frames[idx]
        len = len_0 + len_1 - inter_frame
        ret_seq_0[idx,:,:,:len_0] = seq_comp[idx,:,:,:len_0]
        ret_seq_1[idx,:,:,:len_1] = seq_comp[idx,:,:,len_0-inter_frame:len]

    return ret_seq_0, ret_seq_1