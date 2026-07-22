from utils import add_combined_noise, add_random_noise
from tqdm import tqdm
from utils import r2_score

import numpy as np
import torch

def train_epoch(model, train_loader, criterion, optimizer, device, args=None):
    """
    Train for one epoch.

    Returns: (avg_loss, avg_r2, r2_std)
    """
    model.train()
    running_loss = 0.0
    r2_values = []

    use_noise = args and args.add_noise
    use_clean_skips = getattr(model, '_supports_clean_skips', False)

    fs = getattr(args, 'sampling_freq', 500)
    powerline_freq = getattr(args, 'powerline_freq', 50)

    num_steps = 0

    train_bar = tqdm(train_loader, desc="Training", leave=False)
    for batch in train_bar:
        inputs = batch[0]
        clean_inputs = inputs.to(device)

        pairs = [(clean_inputs, clean_inputs, None)]

        if use_noise:
            random_noise = getattr(args, 'random_noise', False)

            if random_noise:
                noisy_inputs = add_random_noise(
                    clean_inputs,
                    snr_db_min=args.noise_snr_min,
                    snr_db_max=args.noise_snr_max,
                    min_types=getattr(args, 'min_noise_types', 1),
                    max_types=getattr(args, 'max_noise_types', 4),
                    fs=fs,
                    powerline_freq=powerline_freq
                )
            else:
                noise_types = getattr(args, 'noise_types', ['gaussian'])
                noisy_inputs = add_combined_noise(
                    clean_inputs,
                    noise_types=noise_types,
                    snr_db_min=args.noise_snr_min,
                    snr_db_max=args.noise_snr_max,
                    fs=fs,
                    powerline_freq=powerline_freq
                )
            pairs.append((noisy_inputs, clean_inputs, clean_inputs))

        for model_input, target, clean_ref in pairs:
            optimizer.zero_grad()

            if use_clean_skips and clean_ref is not None:
                outputs = model(model_input, x_clean=clean_ref)
            else:
                outputs = model(model_input)

            loss = criterion(outputs, target)
            r2 = r2_score(target, outputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            r2_values.append(r2.item())
            num_steps += 1

        train_bar.set_postfix(loss=f"{loss.item():.4f}", r2=f"{r2.item():.4f}")

    avg_loss = running_loss / num_steps
    avg_r2   = float(np.nanmean(r2_values))
    std_r2   = float(np.nanstd(r2_values))
    return avg_loss, avg_r2, std_r2


def validate(model, val_loader, criterion, device, desc="Validation",
             add_noise=False, noise_snr_min=-5.0, noise_snr_max=15.0,
             noise_types=None, sampling_freq=1000, powerline_freq=60,
             random_noise=False, min_noise_types=1, max_noise_types=4,
             include_clean=True, kl_beta=0.0):
    """
    Validate the model.

    Returns: (avg_loss, avg_r2, r2_std)
    """
    if noise_types is None:
        noise_types = ['gaussian']
    model.eval()
    val_loss = 0.0
    r2_values = []
    num_steps = 0

    val_bar = tqdm(val_loader, desc=desc, leave=False)
    with torch.no_grad():
        for (inputs,) in val_bar:
            clean_inputs = inputs.to(device)

            pairs = []
            if include_clean:
                pairs.append((clean_inputs, clean_inputs))

            if add_noise:
                if random_noise:
                    noisy_inputs = add_random_noise(
                        clean_inputs,
                        snr_db_min=noise_snr_min,
                        snr_db_max=noise_snr_max,
                        min_types=min_noise_types,
                        max_types=max_noise_types,
                        fs=sampling_freq,
                        powerline_freq=powerline_freq
                    )
                else:
                    noisy_inputs = add_combined_noise(
                        clean_inputs,
                        noise_types=noise_types,
                        snr_db_min=noise_snr_min,
                        snr_db_max=noise_snr_max,
                        fs=sampling_freq,
                        powerline_freq=powerline_freq
                    )
                pairs.append((noisy_inputs, clean_inputs))

            if not pairs:
                pairs.append((clean_inputs, clean_inputs))

            for model_input, target in pairs:
                outputs = model(model_input)
                loss = criterion(outputs, target)
                r2 = r2_score(target, outputs)
                val_loss += loss.item()
                r2_values.append(r2.item())
                num_steps += 1

            val_bar.set_postfix(loss=f"{loss.item():.4f}", r2=f"{r2.item():.4f}")

    avg_loss = val_loss / num_steps
    avg_r2   = float(np.nanmean(r2_values))
    std_r2   = float(np.nanstd(r2_values))
    return avg_loss, avg_r2, std_r2
