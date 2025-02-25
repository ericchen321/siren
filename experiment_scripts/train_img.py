# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, meta_modules, utils, training, loss_functions, modules

from torch.utils.data import DataLoader
import configargparse
from functools import partial

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--image_path', type=str, default="data/dog_gt.jpg", help='path to image to be fitted')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
# p.add_argument('--batch_size', type=int, default=1)
p.add_argument('--point_batch_size', type=int, default=500000)
p.add_argument('--eval_patch_size', type=int, default=500000)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
p.add_argument('--num_steps', type=int, default=100000,
               help='Number of epochs to train for.')

p.add_argument('--steps_til_ckpt', type=int, default=10000,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=1000,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options currently are "sine" (all sine activations), "relu" (all relu activations,'
                    '"nerf" (relu activations and positional encoding as in NeRF), "rbf" (input rbf layer, rest relu),'
                    'and in the future: "mixed" (first layer sine, other layers tanh)')
p.add_argument('--num_hidden_layers', type=int, default=3)

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
opt = p.parse_args()

#img_dataset = dataio.Camera()
# Eric: load image from path
img_dataset = dataio.ImageFile(opt.image_path)
img_width, img_height = img_dataset[0].size
img_num_channels = img_dataset.img_channels
image_resolution = (img_height, img_width)
coord_dataset = dataio.Implicit2DWrapper(
    img_dataset, sidelength=image_resolution, compute_diff='none')

print("loading data...")
dataloader = DataLoader(
    coord_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)
print("data loaded.")

# Define the model.
print("defining model...")
if opt.model_type == 'sine' or opt.model_type == 'relu' or opt.model_type == 'tanh' or opt.model_type == 'selu' or opt.model_type == 'elu'\
        or opt.model_type == 'softplus':
    model = modules.SingleBVPNet(
        out_features=img_num_channels,
        type=opt.model_type,
        mode='mlp',
        num_hidden_layers=opt.num_hidden_layers,
        sidelength=image_resolution)
elif opt.model_type == 'rbf' or opt.model_type == 'nerf':
    model = modules.SingleBVPNet(
        out_features=img_num_channels, type='relu', mode=opt.model_type, sidelength=image_resolution)
else:
    raise NotImplementedError
model.cuda()
print("model defined.")

root_path = os.path.join(opt.logging_root, opt.experiment_name)

# Define the loss
loss_fn = partial(loss_functions.image_mse, None)
summary_fn = partial(utils.write_image_summary, image_resolution)

print("starting to train...")
training.train_img(
    model=model, train_dataloader=dataloader, steps=(opt.num_steps+1), lr=opt.lr,
    point_batch_size=opt.point_batch_size, eval_patch_size=opt.eval_patch_size,
    steps_til_summary=opt.steps_til_summary, steps_til_checkpoint=opt.steps_til_ckpt,
    model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn)
print("training done.")
