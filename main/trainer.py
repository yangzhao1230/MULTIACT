import os
import torch
import imageio
import numpy as np
import pickle
# import time
import copy
from torch.optim import AdamW
from torch.nn import DataParallel

# from main.tester import Tester
from data.dataset import get_dataloader
from data.datautils.babel_label import label as BABEL_label

from models.model import get_model

from utils.timer import Timer, sec2minhrs
from utils.logger import colorLogger as Logger
from utils.vis import *
from utils.human_models import SMPLH
from utils.data_split import squence_split, get_cond
from utils.data_split import squence_split, sequence_merge, get_cond, lengths_to_mask
import functools


from diffusion.utils import create_diffusion
from diffusion.resample import create_named_schedule_sampler
from diffusion.resample import LossAwareSampler, UniformSampler

from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d

BABEL_label_rev = {v:k for k, v in BABEL_label.items()}

hyperparams = [
    # overall hyperparams
    "model",

    # training hyperparams
    "batch_size",
    "kl_weight", 
    "train_lr",

    # model hyperparams
    "msg",
]

class Trainer():
    def __init__(self, cfg):

        self.cfg = cfg
        self.action_dataloader = get_dataloader(cfg, "train", cap=cfg.train_per_label)

        self.model = get_model(cfg).cuda()
        print(f"paras total: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        if cfg.model == 'MDM':
            #create diffusiokn
            self.diffusion = create_diffusion(cfg, self.model)
            self.schedule_sampler = create_named_schedule_sampler(cfg.schedule_sampler_type, self.diffusion)


        if cfg.continue_train:
            self.load_model(cfg.load_model_path)
        self.optimizer = AdamW(self.model.parameters(), cfg.train_lr)

        self.train_timer = Timer()
        self.logger = Logger(cfg.log_dir)
        # self.test_logger = Logger(cfg.log_dir, "test.log")

    def train(self):

        self.logger.info("===================================================Training start===================================================")
        # args printing here
        self.logger.info("Using {} action training data".format(len(self.action_dataloader.dataset)))
        self.logger.debug("All settings used:")

        for k in hyperparams:
            self.logger.debug("{}: {}".format(k, vars(self.cfg)[k]))

        self.logger.info("====================================================================================================================")
        self.logger.info("")
        
        self.save_model("init.pkl")
        
        with open(os.path.join(self.cfg.model_out_dir, "sampled_training_data.pkl"), "wb") as f:
            pickle.dump(self.action_dataloader.dataset.dbs[0].sampled, f)
            print("Sampled dataset saved as ", os.path.join(self.cfg.model_out_dir, "sampled_training_data.pkl"))
            setattr(self.cfg, "sampled_data_path", os.path.join(self.cfg.model_out_dir, "sampled_training_data.pkl"))

        self.train_timer.tic()
        for epoch in range(self.cfg.start_epoch, self.cfg.end_epoch):
            self.model.train()
            is_last = (epoch == self.cfg.end_epoch - 1)

            if self.cfg.model == 'MDM':
                epoch_loss = {
                    "loss_s1": torch.tensor(0), 
                    "loss_tr": torch.tensor(0), 
                    "loss_s2": torch.tensor(0)}
            else:
                epoch_loss = {
                    "rec_pose": torch.tensor(0), 
                    "rec_mesh": torch.tensor(0), 
                    "kl_loss": torch.tensor(0),
                    "accel_loss": torch.tensor(0)}

            for itr, inputs in enumerate(self.action_dataloader):
                # forward
                for k, v in inputs.items():
                    inputs[k] = v.to(next(self.model.parameters()).device)
                    # print(v)
                self.optimizer.zero_grad()
                self.model.zero_grad()
                
                if self.cfg.model == 'MDM': # into diffusion
                    # split inputs to 3 parts
                    micro_s1, cond_s1, \
                        micro_tr, cond_tr, \
                        micro_s2, cond_s2 = squence_split(self.cfg, inputs)

                    t, weights = self.schedule_sampler.sample(micro_s1.shape[0], next(self.model.parameters()).device)
                    compute_losses = functools.partial(
                        self.diffusion.training_losses,
                        self.model,
                        # micro,  # [bs, ch, image_size, image_size]
                        t=t,  # [bs](int) sampled timesteps
                        # model_kwargs=micro_cond,
                        # dataset=self.data.dataset
                    )

                        

                    losses_s1 = compute_losses(x_start=micro_s1, model_kwargs=cond_s1)
                    losses_tr = compute_losses(x_start=micro_tr, model_kwargs=cond_tr)
                    losses_s2 = compute_losses(x_start=micro_s2, model_kwargs=cond_s2)

                    loss_s1 = (losses_s1["loss"] * weights).mean()
                    loss_tr = (losses_tr["loss"] * weights).mean()
                    loss_s2 = (losses_s2["loss"] * weights).mean()
                    # log_loss_dict(
                    #     self.diffusion, t, {k: v * weights for k, v in losses.items()}
                    # )
                    # self.mp_trainer.backward(loss)
                    loss = {'loss_s1': loss_s1, 'loss_tr':loss_tr, 'loss_s2':loss_s2}

                else:
                    loss, out = self.model(inputs, 'train')

                    loss = {k: loss[k].mean() for k in loss}

                epoch_loss = {k: loss[k] + epoch_loss[k] for k in loss}

                if self.cfg.model == 'MDM':
                    loss = loss['loss_s1'] + self.cfg.lambda_tr * loss['loss_tr'] + loss['loss_s2']
                else:
                    loss = sum(loss[k] for k in loss)
                
                loss.backward()

                self.optimizer.step()
            
            if (epoch % self.cfg.print_epoch == 0 and epoch != 0) or is_last:
                epoch_loss = {k: epoch_loss[k]/(itr+1) for k in epoch_loss}
                total_time, avg_time = self.train_timer.toc()
                ETA = (avg_time * (self.cfg.end_epoch - self.cfg.start_epoch)) / self.cfg.print_epoch
                ETA = ETA - total_time

                h, m, s = sec2minhrs(ETA)
                h2, m2, s2 = sec2minhrs(total_time)
                print("epoch: {}, avg_time: {} s/epoch, elapsed_time: {} h {} m {} s, ETA: {} h {} m {} s"
                      .format(epoch, round(avg_time / self.cfg.print_epoch, 4), h2, m2, s2, h, m, s))
                self.logger.debug("")
                self.logger.debug("epoch: {}, avg_time: {} s/epoch, elapsed_time: {} h {} m {} s, ETA: {} h {} m {} s"
                    .format(epoch, round(avg_time / self.cfg.print_epoch, 4), h2, m2, s2, h, m, s))
                self.logger.debug(
                    "loss_s1: {}, loss_tr: {}, loss_s2: {}".format(
                        # round(float(epoch_loss["rec_rot"]), 5),
                        round(float(epoch_loss["loss_s1"]), 5),
                        round(float(epoch_loss["loss_tr"]), 5),
                        round(float(epoch_loss["loss_s2"]), 5),
                    ))
            
            if (epoch % self.cfg.vis_epoch == 0 and epoch != 0) or is_last:
                action = inputs["label_mask"][:self.cfg.vis_num]
                valid_mask = inputs["valid_mask"][:self.cfg.vis_num]
                label_texts = [[BABEL_label_rev[int(label_idx)] for label_idx in seq] for seq in action]
                if self.cfg.model == "MDM":
                    vis_inputs = {k: v[:self.cfg.vis_num] for k, v in inputs.items()}
                    out = self.diffusion_sample(vis_inputs)

                gt_mesh = out["gt_smpl_mesh"][:self.cfg.vis_num].detach().cpu()
                gen_mesh = out["gen_smpl_mesh"][:self.cfg.vis_num].detach().cpu()

                names = range(1, self.cfg.vis_num+1)
                names = [str(name) for name in names]
                gt_paths = [os.path.join(self.cfg.vis_dir, "train_vis", str(epoch), "gt") for pth in names]
                gen_paths = [os.path.join(self.cfg.vis_dir, "train_vis", str(epoch), "gen") for pth in names]

                visualize_batch(self.cfg, gen_mesh, valid_mask, gen_paths, names, label_texts, method="render")
                visualize_batch(self.cfg, gt_mesh, valid_mask, gt_paths, names, label_texts, method="render")

                self.save_model("last.pkl")

        return
    
    def diffusion_sample(self, inputs):
        device = next(self.model.parameters()).device
        bs = inputs["smpl_trans"].shape[0]
        hist_frames = self.cfg.hist_frames
        inter_frams = self.cfg.inter_frames
        max_len = self.cfg.max_input_len
        njoints = self.model.njoints
        nfeats = self.model.nfeats

        smpl_trans = inputs["smpl_trans"]
        smpl_param = inputs["smpl_param"]
        label_mask = inputs["label_mask"]
        labels = inputs["labels"]
        frame_length = inputs["frame_length"]

        S1_end_mask = inputs["S1_end_mask"]
        valid_mask = inputs["valid_mask"]
        subaction_mask = inputs["subaction_mask"]
        output_mask = inputs["output_mask"]
        # diffusion sample process
        

        micro_s1, cond_s1, \
            micro_tr, cond_tr, \
            micro_s2, cond_s2 = squence_split(self.cfg, inputs)

        if inter_frams > 0:
            sample_fn = self.diffusion.p_sample_loop_comp

            cond_s1_comp = {}
            cond_s1_comp['y'] = {}
            cond_s1_comp['y']['length'] = [len_s1 + len_tr for (len_s1, len_tr) in zip(cond_s1['y']['length'], cond_tr['y']['length'])]
            cond_s1_comp['y']['mask'] = lengths_to_mask(cond_s1_comp['y']['length'], max_len, device)
            cond_s1_comp['y']['action'] = cond_s1['y']['action']

            cond_s2_comp = {}
            cond_s2_comp['y'] = {}
            cond_s2_comp['y']['length'] = [len_s2 + len_tr for (len_s2, len_tr) in zip(cond_s2['y']['length'], cond_tr['y']['length'])]
            cond_s2_comp['y']['mask'] = lengths_to_mask(cond_s2_comp['y']['length'], max_len, device)
            cond_s2_comp['y']['action'] = cond_s2['y']['action']    

            sample_0, sample_1 = sample_fn(
                self.model,
                cond_tr['y']['length'],
                (bs, njoints, nfeats, max_len),
                (bs, njoints, nfeats, max_len),
                clip_denoised=False,
                model_kwargs_0=cond_s1_comp,
                model_kwargs_1=cond_s2_comp,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=False,
                dump_steps=None,
                noise=None,
                const_noise=False,
                # when experimenting guidance_scale we want to nutrileze the effect of noise on generation
            )

            

        if hist_frames > 0:
            sample_fn = self.diffusion.p_sample_loop
            # cond_s2['y']['length'] = [len_tr + len_s2 for (len_tr, len_s2) in zip(cond_tr['y']['length'], cond_s2['y']['length'])]
            cond_s2['y']['length'] = cond_tr['y']['length'] + cond_s2['y']['length'] + hist_frames
            # cond_s2['y']['length'] = torch.co
            cond_s2['y']['mask'] = lengths_to_mask(cond_s2['y']['length'], max_len + hist_frames, device)
            hist_motion = torch.zeros(bs, njoints, nfeats, hist_frames)
            for idx in range(bs):
                len = cond_s1['y']['length'][idx]
                hist_motion[idx, :, :, :] = micro_s1[idx, :, :, len - hist_frames:len]
            cond_s2['y']['hist_motion'] = hist_motion
            sample_s2 = sample_fn(
                    self.model,
                    (bs, njoints, nfeats, max_len + hist_frames), # FIXME
                    clip_denoised=False,
                    model_kwargs=cond_s2,
                    skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                    init_image=None,
                    progress=False,
                    dump_steps=None,
                    noise=None,
                    const_noise=False,
                )
            sample_s2 = sample_s2[:, :, :, hist_frames:]
            sample_s2 = sample_s2.squeeze().permute(0, 2, 1)
            motion = torch.cat((smpl_param, smpl_trans), dim=2).to(torch.float32).cuda()
            output = copy.deepcopy(motion)

            # tr_s2_mask = subaction_mask > 0
            output[subaction_mask > 0]  = sample_s2[output_mask > 0]
            # cond_tr['y']['length'] += hist_frames
            # cond_tr['y']['mask'] = lengths_to_mask(cond_tr['y']['length'], max_len + hist_frames, next(self.model.parameters()).device)
            # hist_motion = torch.zeros(bs, njoints, nfeats, hist_frames)
            
            # # print(f'batch_size: {bs}')
            # # print(cond_s1['y']['length'])
            # for idx in range(bs):
            #     len = cond_s1['y']['length'][idx]
            #     hist_motion[idx, :, :, :] = micro_s1[idx, :, :, len - hist_frames:len]
            # cond_tr['y']['hist_motion'] = hist_motion
            # # add inpainting_mask and inpainted_motion

            # sample_tr = sample_fn(
            #         self.model,
            #         (bs, njoints, nfeats, max_len + hist_frames),
            #         clip_denoised=False,
            #         model_kwargs=cond_tr,
            #         skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            #         init_image=None,
            #         progress=False,
            #         dump_steps=None,
            #         noise=None,
            #         const_noise=False,
            #     )

            # sample_tr = sample_tr[:, :, :, hist_frames:]

            # cond_s2['y']['length'] += hist_frames
            # cond_s2['y']['mask'] = lengths_to_mask(cond_s2['y']['length'], max_len + hist_frames, next(self.model.parameters()).device)
            # hist_motion = torch.zeros(bs, njoints, nfeats, hist_frames)
            # for idx in range(bs):
            #     len  = cond_tr['y']['length'][idx]
            #     hist_motion[idx, :, :, :] = sample_tr[idx, :, :, len - hist_frames:len]
            # cond_s2['y']['hist_motion'] = hist_motion
            # sample_s2 = sample_fn(
            #         self.model,
            #         (bs, njoints, nfeats, max_len + hist_frames),
            #         clip_denoised=False,
            #         model_kwargs=cond_s2,
            #         skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            #         init_image=None,
            #         progress=False ,
            #         dump_steps=None,
            #         noise=None,
            #         const_noise=False,
            #     )
            
            # sample_s2 = sample_s2[:, :, :, hist_frames:]

            # cond_tr['y']['mask'] = cond_tr['y']['mask'][:, hist_frames:]
            # cond_s2['y']['mask'] = cond_s2['y']['mask'][:, hist_frames:]

            # output = sequence_merge(inputs, sample_tr, cond_tr['y']['mask'], sample_s2, cond_s2['y']['mask'])
        output_param = output[:, :, :156]
        output_trans = output[:, :, 156:]

        smpl_param = smpl_param.to(output_param.device)
        smpl_trans = smpl_trans.to(output_trans.device)

        gt_body_params = {
            "global_orient": smpl_param[:, :, :3],
            "body_pose": smpl_param[:, :, 3:66],
            "hand_pose": smpl_param[:, :, 66:156],
            "transl": smpl_trans
        }

        out_body_params = {
            "global_orient": output_param[:, :, :3],
            "body_pose": output_param[:, :, 3:66],
            "hand_pose": output_param[:, :, 66:156],
            "transl": output_trans
        }

        # smplh_neutral_layer = copy.deepcopy(self.smpl_model.layer).cuda().requires_grad_(False)
        # smplh_joint_regressor = self.smpl_model.joint_regressor.clone().detach().cuda()
        # self.register_buffer("smplh_joint_regressor", smplh_joint_regressor, persistent=False)

        gt_smpl_mesh_cam = self.model.get_batch_smpl_mesh_cam(gt_body_params)
        gt_smpl_mesh_cam = gt_smpl_mesh_cam.reshape(bs, self.cfg.max_input_len, -1, 3)

        out_smpl_mesh_cam = self.model.get_batch_smpl_mesh_cam(out_body_params)
        out_smpl_mesh_cam = out_smpl_mesh_cam.reshape(bs, self.cfg.max_input_len, -1, 3)

        gt_smpl_joint_cam = torch.matmul(self.model.smplh_joint_regressor, gt_smpl_mesh_cam)  # batch * self.cfg.input_duration * self.smpl_model.orig_joint_num * 3, use joint regressor
        out_smpl_joint_cam = torch.matmul(self.model.smplh_joint_regressor, out_smpl_mesh_cam)  # batch * self.cfg.input_duration * self.smpl_model.orig_joint_num * 3, use joint regressor
    

        # generated transition outputs
        trans_out_smpl_joint_cam = torch.zeros_like(out_smpl_joint_cam)
        trans_out_smpl_joint_cam[output_mask==1] = out_smpl_joint_cam[subaction_mask==1]

        # gt transition outputs
        gt_trans_smpl_gt_cam = torch.zeros_like(gt_smpl_joint_cam)
        gt_trans_smpl_gt_cam[output_mask==1] = gt_smpl_joint_cam[subaction_mask==1]

        out = {}

        out['gen_smpl_mesh'] = out_smpl_mesh_cam
        out['gen_smpl_joint'] = out_smpl_joint_cam
        out['gt_smpl_mesh'] = gt_smpl_mesh_cam
        out['gt_smpl_joint'] = gt_smpl_joint_cam
        out['gen_smpl_param'] = output_param
        out['gen_smpl_trans'] = output_trans

        out['gt_transition_joint'] = gt_trans_smpl_gt_cam
        out['gen_transition_joint'] = trans_out_smpl_joint_cam

        return out
    

    def get_state_dict(self, model):
        dump_key = []
        for k in model.state_dict():
            if 'smpl_layer' in k:
                dump_key.append(k)

        update_key = [k for k in model.state_dict().keys() if k not in dump_key]
        return {k: model.state_dict()[k] for k in update_key}

    def save_model(self, name, dump_smpl=True):
        save_path = os.path.join(self.cfg.model_out_dir, name)
        dump_key = []
        if dump_smpl:
            for k in self.model.state_dict():
                if 'smpl_layer' in k:
                    dump_key.append(k)

        update_key = [k for k in self.model.state_dict().keys() if k not in dump_key]
        update_dict = {k: self.model.state_dict()[k] for k in update_key}

        torch.save(update_dict, save_path)
        return

    def visualize_batch(self, meshes, masks, paths, names, labels):
        faces = SMPLH(self.cfg).face
        speed_sec = {'duration': 0.05}

        batch_size, duration = meshes.shape[:2]

        for mesh, mask, path, name, label in zip(meshes, masks, paths, names, labels):

            # vis_img = vis_motion(mesh, faces)
            vis_img = vis_motion_vertices(mesh, faces)

            img_lst = []
            for j in range(duration):
                if int(mask[j]) == 0:
                    continue
                l = label[j]
                i = vis_img[j]
                cv2.putText(i, l, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                img_lst.append(i.astype(np.uint8))

            os.makedirs(path, exist_ok=True)
            imageio.mimsave(os.path.join(path, name+".gif"), img_lst, **speed_sec)
                

    def load_model(self, load_path):
        with open(load_path, "rb") as f:
            data = torch.load(f)

        self.model.load_state_dict(data)
