from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

def main():
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        flash_attn = True
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = 64,
        timesteps = 1000,           # number of steps
        sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    )

    trainer = Trainer(
        diffusion,
        'images/celeba_part',
        train_batch_size = 1,
        train_lr = 8e-5,
        train_num_steps = 10000,         # total training steps
        gradient_accumulate_every = 16,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                       # turn on mixed precision
        calculate_fid = False,              # whether to calculate fid during training
        save_and_sample_every = 500,
        num_samples = 4,
        num_fid_samples = 20,
        results_folder = './results_blonde',
        num_workers = 0,
    )

    trainer.train()
    

if __name__ == '__main__':
    # freeze_support()
    main()