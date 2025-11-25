import torch
import numpy as np
from external.Perspective_and_Equirectangular import e2p
from einops import rearrange, repeat
from torch import nn
import wandb
from lightning.pytorch.utilities import rank_zero_only

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
from ..faed.FAED import FrechetAutoEncoderDistance
from utils.pano import random_sample_camera, horizon_sample_camera
from ..modules.utils import tensor_to_image
from .PanoGenerator import PanoBase



class EvalPanoGen(PanoBase):
    def __init__(
            self,
            log_test_samples: int = 100,
            num_eval_crops: int = 20,
            pano_height: int = 512,
            data: str = None,
            **kwargs,
            ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.eval_metrics = nn.ModuleDict({
            'FID': FrechetInceptionDistance(feature=2048),
            'test_pers_FID': FrechetInceptionDistance(feature=2048),
            'IS': InceptionScore(),
            'KID': KernelInceptionDistance(feature=2048, subsets=50, subset_size=50),
            'rot_FID': FrechetInceptionDistance(feature=2048),
            'rot_IS': InceptionScore(),
            'rot_CS': CLIPScore('openai/clip-vit-base-patch16'),
            'crop_FID': FrechetInceptionDistance(feature=2048, normalize=True),
            'crop_IS': InceptionScore(normalize=True),
            'seam_FID': FrechetInceptionDistance(feature=2048, normalize=True),
            'seam_IS': InceptionScore(normalize=True),
            'mv_FID': FrechetInceptionDistance(feature=2048, normalize=True),
            'mv_IS': InceptionScore(normalize=True),
            'mv_CS': CLIPScore('openai/clip-vit-base-patch16'),
            'test_pers_CS': CLIPScore('openai/clip-vit-base-patch16'),
            'test_pers_CS_no': CLIPScore('openai/clip-vit-base-patch16'),  # Used to determine successful object removal
            'test_pers_KID': KernelInceptionDistance(feature=2048, subsets=50, subset_size=50),
            'FAED': FrechetAutoEncoderDistance(pano_height=pano_height),
            'SSIM': StructuralSimilarityIndexMeasure(data_range=255.0),
            'test_pers_SSIM': StructuralSimilarityIndexMeasure(data_range=255.0),
            'rot_SSIM': StructuralSimilarityIndexMeasure(data_range=255.0),
            'crop_SSIM': StructuralSimilarityIndexMeasure(data_range=1.0),
            'seam_SSIM': StructuralSimilarityIndexMeasure(data_range=1.0),
            'mv_SSIM': StructuralSimilarityIndexMeasure(data_range=1.0),
            'LPIPS': LearnedPerceptualImagePatchSimilarity(normalize=True),
            'test_pers_LPIPS': LearnedPerceptualImagePatchSimilarity(normalize=True),
            # New ERP-specific evaluation metrics
            'PSNR': PeakSignalNoiseRatio(data_range=255.0),
            'L1': MeanAbsoluteError(),
            'L2': MeanSquaredError(),
        })
        self.eval_metrics.requires_grad_(False)

    def load_from_checkpoint(self, *args, **kwargs):
        if 'strict' not in kwargs:
            kwargs['strict'] = False
        return super().load_from_checkpoint(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        if 'strict' not in kwargs:
            kwargs['strict'] = False
        return super().load_state_dict(*args, **kwargs)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        pano_prompt = self.get_pano_prompt(batch)
        test_pers_prompt = batch['test_pers_prompt']
        if 'prompt' in batch:
            pers_prompt = self.get_pers_prompt(batch)

        if batch_idx < self.hparams.log_test_samples:
            test_sample_row = {'pano_id': batch['pano_id'][0] + batch['instruction'][0]}
            test_sample_row['prompt'] = pano_prompt[0]
            test_sample_row['test_pers_prompt'] = test_pers_prompt[0]
            test_sample_row['pano_pred'] = self.temp_wandb_image(tensor_to_image(batch['pano_pred'])[0, 0])
            test_sample_row['pano_gt'] = self.temp_wandb_image(tensor_to_image(batch['pano'])[0, 0])
            test_sample_row['test_pers'] = self.temp_wandb_image(tensor_to_image(batch['test_pers'])[0, 0])
            test_sample_row['ori_test_pers'] = self.temp_wandb_image(tensor_to_image(batch['ori_test_pers'])[0, 0])
            if 'pano_layout_cond' in batch:
                test_sample_row['pano_layout_cond'] = self.temp_wandb_image(
                    tensor_to_image(batch['pano_layout_cond'])[0, 0])

            if batch_idx == 0:
                self.test_sample_table = wandb.Table(columns=list(test_sample_row.keys()))
            self.test_sample_table.add_data(*test_sample_row.values())

        batch = self.trainer.strategy.batch_to_device(batch)
        pano_gt = rearrange(batch['pano'], 'b l c h w -> (b l) c h w')
        pano_gen = rearrange(batch['pano_pred'], 'b l c h w -> (b l) c h w')
        test_pers = rearrange(batch['test_pers'], 'b l c h w -> (b l) c h w')
        ori_test_pers = rearrange(batch['ori_test_pers'], 'b l c h w -> (b l) c h w')
        
            
        self.eval_metrics['FID'].update(pano_gt, real=True)
        self.eval_metrics['FID'].update(pano_gen, real=False)
        self.eval_metrics['test_pers_FID'].update(ori_test_pers, real=True)
        self.eval_metrics['test_pers_FID'].update(test_pers, real=False)
        self.eval_metrics['IS'].update(pano_gen)
        self.eval_metrics['KID'].update(pano_gt, real=True)
        self.eval_metrics['KID'].update(pano_gen, real=False)
        self.eval_metrics['test_pers_KID'].update(ori_test_pers, real=True)
        self.eval_metrics['test_pers_KID'].update(test_pers, real=False)
        # self.eval_metrics['CS'].update(pano_gen.float(), pano_prompt)
        self.eval_metrics['test_pers_CS'].update(test_pers, test_pers_prompt)
        self.eval_metrics['test_pers_CS_no'].update(test_pers, test_pers_prompt)  # Used for successful removal determination
        self.eval_metrics['FAED'].update(pano_gt, real=True)
        self.eval_metrics['FAED'].update(pano_gen, real=False)
        
        # Add SSIM and LPIPS evaluation
        self.eval_metrics['SSIM'].update(pano_gen.float(), pano_gt.float())
        self.eval_metrics['test_pers_SSIM'].update(test_pers.float(), ori_test_pers.float())
        
        # LPIPS requires [0,1] range input
        self.eval_metrics['LPIPS'].update(pano_gen.float()/ 255, pano_gt.float() / 255 )
        self.eval_metrics['test_pers_LPIPS'].update(test_pers.float() / 255, ori_test_pers.float() / 255)
        
        # New ERP-specific evaluation metrics (calculated only on panoramic images)
        self.eval_metrics['PSNR'].update(pano_gen.float(), pano_gt.float())
        self.eval_metrics['L1'].update(pano_gen.float() / 255, pano_gt.float() / 255)
        self.eval_metrics['L2'].update(pano_gen.float() / 255, pano_gt.float() / 255)

        # rotate the panorama and evaluate FID and IS
        pano_gt_rot = torch.roll(pano_gt, pano_gt.shape[3] // 2, 3)
        pano_gen_rot = torch.roll(pano_gen, pano_gen.shape[3] // 2, 3)
        self.eval_metrics['rot_FID'].update(pano_gt_rot, real=True)
        self.eval_metrics['rot_FID'].update(pano_gen_rot, real=False)
        self.eval_metrics['rot_IS'].update(pano_gen_rot)
        # self.eval_metrics['rot_CS'].update(pano_gen_rot, pano_prompt)
        self.eval_metrics['rot_SSIM'].update(pano_gen_rot.float(), pano_gt_rot.float())
        

        # randomly crop the panorama and evaluate FID and IS
        theta, phi = random_sample_camera(self.hparams.num_eval_crops)
        theta, phi = torch.from_numpy(np.rad2deg(theta)).float(), torch.from_numpy(np.rad2deg(phi)).float()
        theta = repeat(theta, 'm -> (b m)', b=batch['pano'].shape[0])
        phi = repeat(phi, 'm -> (b m)', b=batch['pano'].shape[0])
        pano_gt = batch['pano'].expand(-1, self.hparams.num_eval_crops, -1, -1, -1)
        pano_gt = rearrange(pano_gt, 'b m c h w -> (b m) c h w')
        pano_gen = batch['pano_pred'].expand(-1, self.hparams.num_eval_crops, -1, -1, -1)
        pano_gen = rearrange(pano_gen, 'b m c h w -> (b m) c h w')
        pers_gt = e2p(pano_gt.float() / 255, 90, theta, phi, (299, 299))
        pers_gen = e2p(pano_gen.float() / 255, 90, theta, phi, (299, 299))
        # # visulize the cropped panorama
        # pano_gt_sample = pano_gt[0]
        # pers_gt_sample = pers_gt[0]
        # pers_gen_sample = pers_gen[3]
        self.eval_metrics['crop_FID'].update((pers_gt), real=True)
        self.eval_metrics['crop_FID'].update((pers_gen), real=False)
        self.eval_metrics['crop_IS'].update((pers_gen))
        self.eval_metrics['crop_SSIM'].update(pers_gen, pers_gt)

        # crop on the seam and evaluate FID and IS
        theta, phi = random_sample_camera(self.hparams.num_eval_crops)
        theta, phi = torch.from_numpy(np.rad2deg(theta)).float(), torch.from_numpy(np.rad2deg(phi)).float()
        theta[:] = 180
        theta = repeat(theta, 'm -> (b m)', b=batch['pano'].shape[0])
        phi = repeat(phi, 'm -> (b m)', b=batch['pano'].shape[0])
        pano_gt = batch['pano'].expand(-1, self.hparams.num_eval_crops, -1, -1, -1)
        pano_gt = rearrange(pano_gt, 'b m c h w -> (b m) c h w')
        pano_gen = batch['pano_pred'].expand(-1, self.hparams.num_eval_crops, -1, -1, -1)
        pano_gen = rearrange(pano_gen, 'b m c h w -> (b m) c h w')
        pers_gt = e2p(pano_gt.float() / 255, 90, theta, phi, (299, 299))
        pers_gen = e2p(pano_gen.float() / 255, 90, theta, phi, (299, 299))
        # # visulize the cropped panorama
        # pano_gt_sample = pano_gt[0]
        # pers_gt_sample = pers_gt[3]
        # pano_gen_sample = pano_gen[0]
        # pers_gen_sample = pers_gen[3]
        self.eval_metrics['seam_FID'].update((pers_gt ), real=True)
        self.eval_metrics['seam_FID'].update((pers_gen ), real=False)
        self.eval_metrics['seam_IS'].update((pers_gen))
        self.eval_metrics['seam_SSIM'].update(pers_gen, pers_gt)

        # crop as like mvdiffusion and evaluate FID and IS
        theta, phi = horizon_sample_camera(8)
        theta, phi = torch.from_numpy(np.rad2deg(theta)).float(), torch.from_numpy(np.rad2deg(phi)).float()
        theta = repeat(theta, 'm -> (b m)', b=batch['pano'].shape[0])
        phi = repeat(phi, 'm -> (b m)', b=batch['pano'].shape[0])
        pano_gt = batch['pano'].expand(-1, 8, -1, -1, -1)
        pano_gt = rearrange(pano_gt, 'b m c h w -> (b m) c h w')
        pano_gen = batch['pano_pred'].expand(-1, 8, -1, -1, -1)
        pano_gen = rearrange(pano_gen, 'b m c h w -> (b m) c h w')
        pers_gt = e2p(pano_gt.float() / 255, 90, theta, phi, (299, 299))
        pers_gen = e2p(pano_gen.float() / 255, 90, theta, phi, (299, 299))
        # visulize the cropped panorama
        # pano_gt_sample = pano_gt[0]
        # pers_gt_sample = pers_gt[3]
        # pano_gen_sample = pano_gen[0]
        # pers_gen_sample = pers_gen[3]
        self.eval_metrics['mv_FID'].update((pers_gt ), real=True)
        self.eval_metrics['mv_FID'].update((pers_gen ), real=False)
        self.eval_metrics['mv_IS'].update((pers_gen))
        self.eval_metrics['mv_SSIM'].update(pers_gen, pers_gt)


    @torch.no_grad()
    @rank_zero_only
    def on_test_end(self):
        test_metrics = {}
        for key, metric in self.eval_metrics.items():
            if not metric._update_called:
                continue
            if key.endswith('IS') or key.endswith('KID'):
                test_metrics[key], test_metrics[f"{key}_std"] = metric.compute()
            elif key == 'test_pers_CS_no':
                # Use 100-clip_score to determine successful object removal (CLIPScore range is 0-100)
                test_metrics[key] = 100.0 - metric.compute()
            else:
                test_metrics[key] = metric.compute()
        wandb.summary.update(test_metrics)
        metrics_table = wandb.Table(columns=list(test_metrics.keys()), data=[list(test_metrics.values())])
        self.logger.experiment.log({'test/metrics': metrics_table})
        self.logger.experiment.log({'test/sample': self.test_sample_table})
