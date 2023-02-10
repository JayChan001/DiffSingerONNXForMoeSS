import json
import os
import sys
import warnings

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['PYTHONPATH'] = f'"{root_dir}"'
sys.path.insert(0, root_dir)

import argparse
import math
import re
import struct
from functools import partial

import numpy as np
import onnx
import onnxsim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Embedding

from modules.commons.common_layers import Mish
from modules.naive_frontend.encoder import Encoder
from src.diff.diffusion import beta_schedule
from src.diff.net import AttrDict
from utils import load_ckpt
from utils.hparams import hparams, set_hparams
from utils.phoneme_utils import build_phoneme_list
from utils.spk_utils import parse_commandline_spk_mix
from utils.text_encoder import TokenTextEncoder


f0_bin = 256
f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * math.log(1 + f0_min / 700)
f0_mel_max = 1127 * math.log(1 + f0_max / 700)

frozen_spk_embed = None


def f0_to_coarse(f0):
    f0_mel = 1127 * (1 + f0 / 700).log()
    a = (f0_bin - 2) / (f0_mel_max - f0_mel_min)
    b = f0_mel_min * a - 1.
    f0_mel = torch.where(f0_mel > 0, f0_mel * a - b, f0_mel)
    torch.clip_(f0_mel, min=1., max=float(f0_bin - 1))
    f0_coarse = torch.round(f0_mel).long()
    return f0_coarse


class LengthRegulator(nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, dur):
        token_idx = torch.arange(1, dur.shape[1] + 1, device=dur.device)[None, :, None]
        dur_cumsum = torch.cumsum(dur, dim=1)
        dur_cumsum_prev = F.pad(dur_cumsum, (1, -1), mode='constant', value=0)
        pos_idx = torch.arange(dur.sum(dim=1).max(), device=dur.device)[None, None]
        token_mask = (pos_idx >= dur_cumsum_prev[:, :, None]) & (pos_idx < dur_cumsum[:, :, None])
        mel2ph = (token_idx * token_mask).sum(dim=1)
        return mel2ph


class FastSpeech2MIDILess(nn.Module):
    def __init__(self, dictionary):
        super().__init__()
        self.lr = LengthRegulator()
        self.txt_embed = Embedding(len(dictionary), hparams['hidden_size'], dictionary.pad())
        self.dur_embed = Linear(1, hparams['hidden_size'])
        self.encoder = Encoder(self.txt_embed, hparams['hidden_size'], hparams['enc_layers'],
                               hparams['enc_ffn_kernel_size'], num_heads=hparams['num_heads'])
        self.f0_embed_type = hparams.get('f0_embed_type', 'discrete')
        if self.f0_embed_type == 'discrete':
            self.pitch_embed = Embedding(300, hparams['hidden_size'], dictionary.pad())
        elif self.f0_embed_type == 'continuous':
            self.pitch_embed = Linear(1, hparams['hidden_size'])
        else:
            raise ValueError('f0_embed_type must be \'discrete\' or \'continuous\'.')
        if hparams['use_spk_id']:
            self.spk_embed_proj = Embedding(hparams['num_spk'], hparams['hidden_size'])
        self.hasspk = hparams['use_spk_id']

    def forward(self, tokens, durations, f0, spk_embed=None):
        if self.hasspk:
            spk_embed_id = spk_embed
            spk_embed_dur_id = spk_embed_id
            spk_embed_f0_id = spk_embed_id
            spk_embed = self.spk_embed_proj(spk_embed_id)[:, None, :]
            spk_embed_dur = spk_embed_f0 = spk_embed
        else:
            spk_embed = spk_embed[0]
        durations *= tokens > 0
        mel2ph = self.lr.forward(durations)
        f0 *= mel2ph > 0
        mel2ph = mel2ph[..., None].repeat((1, 1, hparams['hidden_size']))
        dur_embed = self.dur_embed(durations.float()[:, :, None])
        encoded = self.encoder(tokens, dur_embed)
        encoded = F.pad(encoded, (0, 0, 1, 0))
        encoded = torch.gather(encoded, 1, mel2ph)
        if self.f0_embed_type == 'discrete':
            pitch = f0_to_coarse(f0)
            pitch_embed = self.pitch_embed(pitch)
        else:
            f0_mel = (1 + f0 / 700).log()
            pitch_embed = self.pitch_embed(f0_mel[:, :, None])
        condition = encoded + pitch_embed
        condition += spk_embed
        return condition.transpose(1, 2)


def extract(a, t):
    return a[t].reshape((1, 1, 1, 1))


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        self.register_buffer('emb', torch.exp(torch.arange(half_dim) * torch.tensor(-emb)).unsqueeze(0))

    def forward(self, x):
        emb = self.emb * x
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, encoder_hidden, residual_channels, dilation):
        super().__init__()
        self.residual_channels = residual_channels
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            3,
            padding=dilation,
            dilation=dilation)
        self.diffusion_projection = Linear(residual_channels, residual_channels)
        self.conditioner_projection = nn.Conv1d(encoder_hidden, 2 * residual_channels, 1)
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)
        y = x + diffusion_step

        y = self.dilated_conv(y) + conditioner

        # Using torch.split instead of torch.chunk to avoid using onnx::Slice
        gate, filter = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)

        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(y)

        # Using torch.split instead of torch.chunk to avoid using onnx::Slice
        residual, skip = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)

        return (x + residual) / math.sqrt(2.0), skip


class DiffNet(nn.Module):
    def __init__(self, in_dims=80):
        super().__init__()
        self.params = params = AttrDict(
            # Model params
            encoder_hidden=hparams['hidden_size'],
            residual_layers=hparams['residual_layers'],
            residual_channels=hparams['residual_channels'],
            dilation_cycle_length=hparams['dilation_cycle_length'],
        )
        self.input_projection = nn.Conv1d(in_dims, params.residual_channels, 1)
        self.diffusion_embedding = SinusoidalPosEmb(params.residual_channels)
        dim = params.residual_channels
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )
        self.residual_layers = nn.ModuleList([
            ResidualBlock(params.encoder_hidden, params.residual_channels, 2 ** (i % params.dilation_cycle_length))
            for i in range(params.residual_layers)
        ])
        self.skip_projection = nn.Conv1d(params.residual_channels, params.residual_channels, 1)
        self.output_projection = nn.Conv1d(params.residual_channels, in_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond):
        """
        :param spec: [B, 1, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, M, T]
        :return:
        """
        x = spec.squeeze(1)
        x = self.input_projection(x)  # [B, residual_channel, T]

        x = F.relu(x)
        diffusion_step = diffusion_step.float()
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)  # [B, mel_bins, T]
        return x.unsqueeze(1)


class NaiveNoisePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('clip_min', to_torch(-1.))
        self.register_buffer('clip_max', to_torch(1.))

    def forward(self, x, noise_pred, t):
        x_recon = (
                extract(self.sqrt_recip_alphas_cumprod, t) * x -
                extract(self.sqrt_recipm1_alphas_cumprod, t) * noise_pred
        )
        x_recon = torch.clamp(x_recon, min=self.clip_min, max=self.clip_max)

        model_mean = (
                extract(self.posterior_mean_coef1, t) * x_recon +
                extract(self.posterior_mean_coef2, t) * x
        )
        model_log_variance = extract(self.posterior_log_variance_clipped, t)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = ((t > 0).float()).reshape(1, 1, 1, 1)
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise


class PLMSNoisePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        to_torch = partial(torch.tensor, dtype=torch.float32)

        # Below are buffers for TorchScript to pass jit compilation.
        self.register_buffer('_1', to_torch(1))
        self.register_buffer('_2', to_torch(2))
        self.register_buffer('_3', to_torch(3))
        self.register_buffer('_5', to_torch(5))
        self.register_buffer('_9', to_torch(9))
        self.register_buffer('_12', to_torch(12))
        self.register_buffer('_16', to_torch(16))
        self.register_buffer('_23', to_torch(23))
        self.register_buffer('_24', to_torch(24))
        self.register_buffer('_37', to_torch(37))
        self.register_buffer('_55', to_torch(55))
        self.register_buffer('_59', to_torch(59))

    def forward(self, x, noise_t, t, t_prev):
        a_t = extract(self.alphas_cumprod, t)
        a_prev = extract(self.alphas_cumprod, t_prev)
        a_t_sq, a_prev_sq = a_t.sqrt(), a_prev.sqrt()

        x_delta = (a_prev - a_t) * ((self._1 / (a_t_sq * (a_t_sq + a_prev_sq))) * x - self._1 / (
                a_t_sq * (((self._1 - a_prev) * a_t).sqrt() + ((self._1 - a_t) * a_prev).sqrt())) * noise_t)
        x_pred = x + x_delta

        return x_pred

    def predict_stage0(self, noise_pred, noise_pred_prev):
        return (noise_pred
                + noise_pred_prev) / self._2

    def predict_stage1(self, noise_pred, noise_list):
        return (noise_pred * self._3
                - noise_list[-1]) / self._2

    def predict_stage2(self, noise_pred, noise_list):
        return (noise_pred * self._23
                - noise_list[-1] * self._16
                + noise_list[-2] * self._5) / self._12

    def predict_stage3(self, noise_pred, noise_list):
        return (noise_pred * self._55
                - noise_list[-1] * self._59
                + noise_list[-2] * self._37
                - noise_list[-3] * self._9) / self._24


class MelExtractor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.squeeze(1).permute(0, 2, 1)
        d = (self.spec_max - self.spec_min) / 2
        m = (self.spec_max + self.spec_min) / 2
        mel_out = x * d.cuda() + m.cuda()
        return mel_out


class GaussianDiffusion(nn.Module):
    def __init__(self, out_dims, timesteps=1000, k_step=1000, spec_min=None, spec_max=None):
        super().__init__()
        self.mel_bins = out_dims
        self.K_step = k_step

        self.denoise_fn = DiffNet(out_dims)
        self.naive_noise_predictor = NaiveNoisePredictor()
        self.plms_noise_predictor = PLMSNoisePredictor()
        self.mel_extractor = MelExtractor()

        betas = beta_schedule[hparams.get('schedule_type', 'cosine')](timesteps)

        # Below are buffers for state_dict to load into.
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        self.register_buffer('spec_min', torch.FloatTensor(spec_min)[None, None, :hparams['keep_bins']])
        self.register_buffer('spec_max', torch.FloatTensor(spec_max)[None, None, :hparams['keep_bins']])

    def build_submodules(self):
        # Move registered buffers into submodules after loading state dict.
        self.naive_noise_predictor.register_buffer('sqrt_recip_alphas_cumprod', self.sqrt_recip_alphas_cumprod)
        self.naive_noise_predictor.register_buffer('sqrt_recipm1_alphas_cumprod', self.sqrt_recipm1_alphas_cumprod)
        self.naive_noise_predictor.register_buffer(
            'posterior_log_variance_clipped', self.posterior_log_variance_clipped)
        self.naive_noise_predictor.register_buffer('posterior_mean_coef1', self.posterior_mean_coef1)
        self.naive_noise_predictor.register_buffer('posterior_mean_coef2', self.posterior_mean_coef2)
        self.plms_noise_predictor.register_buffer('alphas_cumprod', self.alphas_cumprod)
        self.mel_extractor.register_buffer('spec_min', self.spec_min)
        self.mel_extractor.register_buffer('spec_max', self.spec_max)
        del self.sqrt_recip_alphas_cumprod
        del self.sqrt_recipm1_alphas_cumprod
        del self.posterior_log_variance_clipped
        del self.posterior_mean_coef1
        del self.posterior_mean_coef2
        del self.alphas_cumprod
        del self.spec_min
        del self.spec_max

    def forward(self, condition, speedup, Onnx, project_name):
        condition = condition.transpose(1, 2)  # (1, n_frames, 256) => (1, 256, n_frames)

        device = condition.device
        n_frames = condition.shape[2]
        step_range = torch.arange(0, self.K_step, speedup, dtype=torch.long, device=device).flip(0)
        x = torch.randn((1, 1, self.mel_bins, n_frames), device=device)

        if speedup > 1:
            if Onnx:
                ot = step_range[0]
                ot_1 = torch.full((1,), ot, device=device, dtype=torch.long)
                torch.onnx.export(
                    self.denoise_fn,
                    (x.cuda(), ot_1.cuda(), condition.cuda()),
                    f"onnx/{project_name}_denoise.onnx",
                    input_names=["noise", "time", "condition"],
                    output_names=["noise_pred"],
                    dynamic_axes={
                        "noise": [3],
                        "condition": [2]
                    },
                    opset_version=16
                )
            plms_noise_stage = torch.tensor(0, dtype=torch.long, device=device)
            noise_list = torch.zeros((0, 1, 1, self.mel_bins, n_frames), device=device)
            for t in step_range:
                noise_pred = self.denoise_fn(x, t, condition)
                t_prev = t - speedup
                t_prev = t_prev * (t_prev > 0)

                if plms_noise_stage == 0:
                    if Onnx:
                        torch.onnx.export(
                            self.plms_noise_predictor,
                            (x.cuda(), noise_pred.cuda(), t.cuda(), t_prev.cuda()),
                            f"onnx/{project_name}_pred.onnx",
                            input_names=["noise", "noise_pred", "time", "time_prev"],
                            output_names=["noise_pred_o"],
                            dynamic_axes={
                                "noise": [3],
                                "noise_pred": [3]
                            },
                            opset_version=16
                        )
                    x_pred = self.plms_noise_predictor(x, noise_pred, t, t_prev)
                    noise_pred_prev = self.denoise_fn(x_pred, t_prev, condition)
                    noise_pred_prime = self.plms_noise_predictor.predict_stage0(noise_pred, noise_pred_prev)
                elif plms_noise_stage == 1:
                    noise_pred_prime = self.plms_noise_predictor.predict_stage1(noise_pred, noise_list)
                elif plms_noise_stage == 2:
                    noise_pred_prime = self.plms_noise_predictor.predict_stage2(noise_pred, noise_list)
                else:
                    noise_pred_prime = self.plms_noise_predictor.predict_stage3(noise_pred, noise_list)

                noise_pred = noise_pred.unsqueeze(0)
                if plms_noise_stage < 3:
                    noise_list = torch.cat((noise_list, noise_pred), dim=0)
                    plms_noise_stage = plms_noise_stage + 1
                else:
                    noise_list = torch.cat((noise_list[-2:], noise_pred), dim=0)
                x = self.plms_noise_predictor(x, noise_pred_prime, t, t_prev)
        else:
            for t in step_range:
                pred = self.denoise_fn(x, t, condition)
                x = self.naive_noise_predictor(x, pred, t)
        if Onnx:
            torch.onnx.export(
                self.mel_extractor,
                x.cuda(),
                f"onnx/{project_name}_after.onnx",
                input_names=["x"],
                output_names=["mel_out"],
                dynamic_axes={
                    "x": [3]
                },
                opset_version=16
            )
        mel = self.mel_extractor(x)
        return mel


def build_fs2_model(device):
    model = FastSpeech2MIDILess(
        dictionary=TokenTextEncoder(vocab_list=build_phoneme_list())
    )
    model.eval()
    load_ckpt(model, hparams['work_dir'], 'model.fs2', strict=True)
    model.to(device)
    return model


def build_diff_model(device):
    model = GaussianDiffusion(
        out_dims=hparams['audio_num_mel_bins'],
        timesteps=hparams['timesteps'],
        k_step=hparams['K_step'],
        spec_min=hparams['spec_min'],
        spec_max=hparams['spec_max'],
    )
    model.eval()
    load_ckpt(model, hparams['work_dir'], 'model', strict=False)
    model.build_submodules()
    model.to(device)
    return model


class ModuleWrapper(nn.Module):
    def __init__(self, model, name='model'):
        super().__init__()
        self.wrapped_name = name
        setattr(self, name, model)

    def forward(self, *args, **kwargs):
        return getattr(self, self.wrapped_name)(*args, **kwargs)


class FastSpeech2Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = ModuleWrapper(model, name='fs2')

    def forward(self, tokens, durations, f0, spk_embed=None):
        return self.model(tokens, durations, f0, spk_embed=spk_embed)


class DiffusionWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, condition, speedup,Onnx,project_name):
        return self.model(condition, speedup, Onnx=Onnx, project_name=project_name)

def export(name=None,Onnx=False):
    set_hparams(print_hparams=False)
    if hparams.get('use_midi', True) or not hparams['use_pitch_embed']:
        raise NotImplementedError('Only checkpoints of MIDI-less mode are supported.')

    # Build models to export
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fs2 = FastSpeech2Wrapper(
        model=build_fs2_model(device)
    )
    diffusion = DiffusionWrapper(
        model=build_diff_model(device)
    )

    with torch.no_grad():
        # Export speakers and speaker mixes
        n_frames = 10
        tokens = torch.tensor([[3]], dtype=torch.long, device=device)
        durations = torch.tensor([[n_frames]], dtype=torch.long, device=device)
        f0 = torch.tensor([[440.] * n_frames], dtype=torch.float32, device=device)
        spk_embed = torch.LongTensor([0])
        kwargs = {}
        arguments = (tokens, durations, f0, spk_embed)
        input_names = ['tokens', 'durations', 'f0', 'spk_embed']
        dynamix_axes = {
            'tokens': {
                1: 'n_tokens'
            },
            'durations': {
                1: 'n_tokens'
            },
            'f0': {
                1: 'n_frames'
            }
        }
        print('Exporting Encoder...')
        if Onnx:
            torch.onnx.export(
                fs2,
                arguments,
                f"onnx/{name}_encoder.onnx",
                input_names=input_names,
                output_names=[
                    'condition'
                ],
                dynamic_axes=dynamix_axes,
                opset_version=11
            )
        shape = (1, 1, hparams['audio_num_mel_bins'], n_frames)
        condition = torch.rand((1, n_frames, hparams['hidden_size']), device=device)
        speedup = torch.tensor(10, dtype=torch.long, device=device)
        dummy = diffusion.forward(condition, speedup,Onnx=Onnx,project_name=name)

        print('PyTorch ONNX export finished.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export DiffSinger acoustic model to ONNX format.')
    parser.add_argument('--target', required=False, type=str, help='path of the target ONNX model')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--spk', required=False, type=str, action='append',
                        help='(for combined models) speakers or speaker mixes to export')
    group.add_argument('--freeze_spk', required=False, type=str,
                        help='(for combined models) speaker or speaker mix to freeze into the model')
    parser.add_argument('--out', required=False, type=str, help='output directory for ONNX models and speaker keys')
    args = parser.parse_args()

    # Deprecation for --target argument
    assert args.target is None, 'The \'--target\' argument is deprecated. ' \
                                'Please use the \'--out\' argument to specified the output directory if needed.'

    # Temporarily disable --spk argument
    if args.spk is not None:
        raise NotImplementedError('Exporting speakers or speaker mixes is not supported yet.')

    exp = "utagoe"
    cwd = os.getcwd()
    if args.out:
        out = os.path.join(cwd, args.out) if not os.path.isabs(args.out) else args.out
    else:
        out = f'onnx/assets/{exp}'
    os.chdir(root_dir)
    sys.argv = [
        'inference/ds_cascade.py',
        '--exp_name',
        exp,
        '--infer'
    ]

    os.makedirs(f'onnx/temp', exist_ok=True)
    diff_model_path = f'onnx/temp/diffusion.onnx'
    fs2_model_path = f'onnx/temp/fs2.onnx'
    spk_name_pattern = r'[0-9A-Za-z_-]+'
    spk_export_paths = None
    frozen_spk_name = None
    frozen_spk_mix = None
    if args.spk is not None:
        spk_export_paths = []
        for spk_export in args.spk:
            assert '=' in spk_export or '|' not in spk_export, \
                'You must specify an alias with \'NAME=\' for each speaker mix.'
            if '=' in spk_export:
                alias, mix = spk_export.split('=', maxsplit=1)
                assert re.fullmatch(spk_name_pattern, alias) is not None, f'Invalid alias \'{alias}\' for speaker mix.'
                spk_export_paths.append({'mix': mix, 'path': f'onnx/assets/temp/{alias}.npy'})
            else:
                assert re.fullmatch(spk_name_pattern, spk_export) is not None, \
                    f'Invalid alias \'{spk_export}\' for speaker mix.'
                spk_export_paths.append({'mix': spk_export, 'path': f'onnx/assets/temp/{spk_export}.npy'})
    elif args.freeze_spk is not None:
        assert '=' in args.freeze_spk or '|' not in args.freeze_spk, \
            'You must specify an alias with \'NAME=\' for each speaker mix.'
        if '=' in args.freeze_spk:
            alias, mix = args.freeze_spk.split('=', maxsplit=1)
            assert re.fullmatch(spk_name_pattern, alias) is not None, f'Invalid alias \'{alias}\' for speaker mix.'
            frozen_spk_name = alias
            frozen_spk_mix = mix
        else:
            assert re.fullmatch(spk_name_pattern, args.freeze_spk) is not None, \
                f'Invalid alias \'{args.freeze_spk}\' for speaker mix.'
            frozen_spk_name = args.freeze_spk
            frozen_spk_mix = args.freeze_spk

    if frozen_spk_name is None:
        target_model_path = f'{out}/{exp}.onnx'
    else:
        target_model_path = f'{out}/{exp}.{frozen_spk_name}.onnx'
    os.makedirs(out, exist_ok=True)
    export(name=exp,Onnx=True)
    
