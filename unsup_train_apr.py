import torch
import numpy as np
from cpe.datasets import BlenderDataset
from cpe.cpe_nerf_utils import load_nerf_for_cpe, render_poses_for_cpe, render_for_cpe
from run_nerf_helpers import get_rays
import torchvision.transforms as transforms
from cpe.pose_regressors import APR, RPR
import torch.nn as nn


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # CPE training options
    # Place to save the checkpoint
    parser.add_argument("--num_epochs", type=int, default=10,
                       help='number of epochs')
    parser.add_argument("--lr", type=float, default=1e-04,
                        help='learning rate')
    parser.add_argument("--weight_decay", type=float, default=1e-04,
                        help='weight decay')

    # NeRF architecture options
    parser.add_argument("--nerf_ckpt_path", type=str, default='./logs/',
                        help='path to pretrained nerf')
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')


    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()

    # Set seeds
    np.random.seed(0)
    #TODO cuda seeds

    # Create the data
    # TODO add support for other datasets
    dataset = BlenderDataset(args.datadir, split='train')
    loader_params = {'batch_size': 4,
                     'shuffle': True,
                     'num_workers': 4}
    data_loader = torch.utils.data.DataLoader(dataset, **loader_params)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load nerf model for supervising CPE
    render_kwargs = load_nerf_for_cpe(args.nerf_ckpt_path, device, multires=args.multires, i_embed=args.i_embed,
                                      use_viewdirs=dataset.dataset_flags.get('use_viewdirs'), multires_views=args.multires_views,
                                      N_importance=128, netwidth=256, netdepth=8,
                                      netwidth_fine=256, netdepth_fine=8, netchunk=1024*64,
                                      white_bkgd=dataset.dataset_flags.get('white_bkgd'),
                                      N_samples=args.N_samples, no_ndc=dataset.dataset_flags.get('no_ndc'))
    render_kwargs.update(dataset.bds)


    # Create the regressors, optimizer and loss
    apr = APR().to(device)
    rpr = RPR().to(device)
    params = list(apr.parameters()) + list(rpr.parameters())
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                             lr=args.lr,
                             weight_decay=args.weight_decay)
    criterion = nn.MSELoss().to(device)
    normalize_transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    # Training camera pose estimation without pose labels
    for epoch in range(args.num_epochs):
        for idx, sample in enumerate(data_loader, 0):
            print(idx)

            # Load images and intrinsic
            imgs = sample.get('torch_img').to(device)
            hwf = sample.get('hwf').to(device)
            K = sample.get('K').to(device)

            optimizer.zero_grad()
            # Initial pose estimation with absolute pose regression
            # The APR also predicts the positions to use
            poses_from_apr, positions = apr(imgs)

            # Render the images according to the estimated poses
            rgbs_from_apr = render_poses_for_cpe(poses_from_apr, hwf, K, args.chunk, render_kwargs, gt_imgs=None,
                                      savedir=None, render_factor=0)[0]

            # Normalize the output images and compute the MSE loss
            rgbs_from_apr = normalize_transform(rgbs_from_apr)
            img_loss_apr = criterion(rgbs_from_apr, imgs)

            # Use the positions and estimated poses from the APR to perform relative pose regression
            # Note that this is a much cheaper operation than rendering the entire image, which we can leverage at test time
            poses_from_rpr = rpr(rgbs_from_apr[positions],imgs[positions], poses_from_apr)
            rgbs_from_rpr = []
            for i, pose in enumerate(poses_from_rpr):
                rays_o, rays_d = get_rays(hwf[0], hwf[1], hwf[2], pose)  # (H, W, 3), (H, W, 3)
                rays_o = rays_o[positions[i, :, 1], positions[i, :, 0]]
                rays_d = rays_d[positions[i, :, 1], positions[i, :, 0]]
                batch_rays = torch.stack([rays_o, rays_d], 0)
                rgbs_from_rpr = render_for_cpe(hwf[1], hwf[2], K, rays=batch_rays, **render_kwargs)[0]

            # Normalize the output pixels and compute the MSE loss
            rgbs_from_apr = normalize_transform(rgbs_from_rpr)
            img_loss_rpr = criterion(rgbs_from_apr, imgs)

            # Backprop with the total loss
            loss = img_loss_apr + img_loss_rpr
            loss.backward()
            optimizer.step()


if __name__=='__main__':
    train()
