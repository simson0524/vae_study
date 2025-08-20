# src/visualization.py

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3D projection
import numpy as np
import torch
import matplotlib.pyplot as plt

def collect_latents(model, loader, device="cuda", max_points=1024):
    """
    Run the model over the loader and collect up to `max_points` latent vectors
    (using the *reparameterized* latent z), along with optional labels if provided.
    """
    model.eval()
    latents = []
    labels = []
    points_collected = 0
    device = torch.device(device)

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                x = batch[0].to(device)
                y = batch[1] if len(batch) > 1 else None
            else:
                x = batch.to(device)
                y = None

            reconst_x, mu, logvar, latent_z = model(x)
            z = latent_z.detach().cpu().numpy()
            latents.append(z)

            if y is not None:
                labels.append(np.array(y))

            points_collected += z.shape[0]
            if points_collected >= max_points:
                break

    Z = np.concatenate(latents, axis=0)
    if labels:
        Y = np.concatenate(labels, axis=0)
    else:
        Y = None
    return Z[:max_points], Y[:max_points] if Y is not None else None


def ensure_3d(Z):
    """
    Ensure Z has 3 dimensions for plotting. If latent_dim >= 3, take first 3 dims.
    Otherwise, pad or project with PCA to 3D.
    """
    from sklearn.decomposition import PCA

    if Z.shape[1] >= 3:
        return Z[:, :3]
    else:
        # If latent dim < 3, apply PCA to get 3D (even if from 1D/2D)
        pca = PCA(n_components=3)
        Z3 = pca.fit_transform(Z)
        return Z3
    

def _kde_density_per_class(Z3, Y, bandwidth=0.6):
    try:
        from sklearn.neighbors import KernelDensity
    except Exception as e:
        print("scikit-learn is required for KDE-based density. Install with `pip install scikit-learn`.")
        raise e

    dens = np.zeros(Z3.shape[0], dtype=np.float32)
    classes = np.unique(Y)
    for cls in classes:
        mask = (Y == cls)
        if mask.sum() < 2:
            # Too few points to estimate density; set minimal constant
            dens[mask] = 0.0
            continue
        kde = KernelDensity(bandwidth=bandwidth)
        kde.fit(Z3[mask])
        log_d = kde.score_samples(Z3[mask])
        d = np.exp(log_d)
        # Normalize within class to [0,1] for fair visual scaling per class
        d_min, d_max = d.min(), d.max()
        if d_max > d_min:
            d = (d - d_min) / (d_max - d_min)
        else:
            d = np.zeros_like(d)
        dens[mask] = d.astype(np.float32)
    return dens


def plot_latent_3d_cloud_by_class(Z3, Y, bandwidth=0.6, s_min=3, s_max=20, alpha=0.15, title="Latent space (density cloud by class)"):
    """
    Plot 3D latent points, class by class, with point size reflecting per-class local density.
    - Colors: Matplotlib default cycle (per class), not explicitly set.
    - Size: interpolated from density in [s_min, s_max].
    - Alpha: constant (overlap + size conveys cloud density visually).
    """
    if Y is None:
        raise ValueError("Y (labels) are required for class-wise density clouds.")

    dens = _kde_density_per_class(Z3, Y, bandwidth=bandwidth)
    sizes = s_min + dens * (s_max - s_min)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    classes = np.unique(Y)

    for cls in classes:
        mask = (Y == cls)
        # Plot each class separately so it receives a distinct default color
        ax.scatter(
            Z3[mask, 0], Z3[mask, 1], Z3[mask, 2],
            s=sizes[mask],
            alpha=alpha,
            depthshade=False,
            label=str(cls),
        )

    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_zlabel("z3")
    ax.set_title(title)
    ax.legend(title="Class")
    plt.show()


def show_recon_grid_with_z(model, loader, device="cuda", num_images=16, title="Original vs Reconstruction (+ latent_z)"):
    model.eval()
    device = torch.device(device)

    batch = next(iter(loader))
    if isinstance(batch, (list, tuple)) and len(batch) >= 1:
        x = batch[0].to(device)
    else:
        x = batch.to(device)

    with torch.no_grad():
        reconst_x, mu, logvar, latent_z = model(x)

    x = x.detach().cpu()
    rx = reconst_x.detach().cpu()
    z = latent_z.detach().cpu().numpy()

    n = min(num_images, x.size(0))

    def to_numpy(img_tensor):
        arr = img_tensor.numpy()
        arr = np.clip(arr, 0.0, 1.0)
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        return arr

    fig, axes = plt.subplots(n, 2, figsize=(7.2, 3.6*n), squeeze=False)
    fig.suptitle(title)

    for i in range(n):
        axes[i, 0].imshow(to_numpy(x[i]))
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f"Origin {i}\nz : {latent_z[i]}")

        axes[i, 1].imshow(to_numpy(rx[i]))
        axes[i, 1].axis('off')
        axes[i, 1].set_title(f"Reconst {i}")

    plt.tight_layout()
    plt.show()


def visualize_vae_cloud(model, loader, device="cuda", latent_max_points=10000,
                        bandwidth=0.6, s_min=3, s_max=20, alpha=0.15, recon_images=16):
    try:
        Z, Y = collect_latents(model, loader, device=device, max_points=latent_max_points)
        Z3 = ensure_3d(Z)
    except NameError as e:
        print("Missing helper(s) `collect_latents` or `ensure_3d`. Please define them first.")
        raise e

    plot_latent_3d_cloud_by_class(Z3, Y, bandwidth=bandwidth, s_min=s_min, s_max=s_max, alpha=alpha,
                                  title="ConvVAE latent_z (3D) — density cloud per class")
    show_recon_grid_with_z(model, loader, device=device, num_images=recon_images,
                           title="Original (top) vs Reconstruction (bottom) — with latent_z per sample")