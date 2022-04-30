'''Implements a generic training loop.
'''

import torch
import utils
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import shutil
import gc


def train(model, dataset, train_dataloader, epochs, lr, eval_patch_size, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          summary_fn, val_dataloader=None, double_precision=False, clip_grad=False, use_lbfgs=False, loss_schedules=None):

    optim = torch.optim.Adam(lr=lr, params=model.parameters())

    # copy settings from Raissi et al. (2019) and here 
    # https://github.com/maziarraissi/PINNs
    if use_lbfgs:
        optim = torch.optim.LBFGS(lr=lr, params=model.parameters(), max_iter=50000, max_eval=50000,
                                  history_size=50, line_search_fn='strong_wolfe')

    if os.path.exists(model_dir):
        val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
        if val == 'y':
            shutil.rmtree(model_dir)

    os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    total_steps = 0
    model_input_eval, gt_eval = dataset.get_item_for_eval()
    model_input_eval_arr = torch.split(model_input_eval['coords'], eval_patch_size, dim=1)
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            if not epoch % epochs_til_checkpoint and epoch:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                           np.array(train_losses))

            for step, (model_input_train, gt_train) in enumerate(train_dataloader):
                start_time = time.time()
            
                model_input_train = {key: value.cuda() for key, value in model_input_train.items()}
                gt_train = {key: value.cuda() for key, value in gt_train.items()}

                if double_precision:
                    model_input_train = {key: value.double() for key, value in model_input_train.items()}
                    gt_train = {key: value.double() for key, value in gt_train.items()}

                if use_lbfgs:
                    def closure():
                        optim.zero_grad()
                        model_output_train = model(model_input_train)
                        losses = loss_fn(model_output_train, gt_train)
                        train_loss = 0.
                        for loss_name, loss in losses.items():
                            train_loss += loss.mean() 
                        train_loss.backward()
                        return train_loss
                    optim.step(closure)

                # print(f"forward pass, {model_input_train['coords'].shape}")
                model_output_train = model(model_input_train)
                losses = loss_fn(model_output_train, gt_train)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if loss_schedules is not None and loss_name in loss_schedules:
                        writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                        single_loss *= loss_schedules[loss_name](total_steps)

                    writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss

                train_losses.append(train_loss.item())
                writer.add_scalar("total_train_loss", train_loss, total_steps)

                if not total_steps % steps_til_summary:
                    # for evaluation, we use all points as input to the model and we feed these
                    # points in patches
                    model_out_arrs = []
                    for patch_index in range(len(model_input_eval_arr)):
                        model_input_eval_patch = {
                            'coords': model_input_eval_arr[patch_index]
                        }
                        model.eval()
                        model_output_eval_patch = None
                        with torch.no_grad():
                            model_output_eval_patch = model(model_input_eval_patch)
                        model.train()
                        model_out_arrs.append(model_output_eval_patch['model_out'])
                    model_output_eval = {
                        'model_in': model_input_eval['coords'],
                        'model_out': torch.cat(model_out_arrs, dim=1).cuda()
                    }
                    gt_eval = {key: value.cuda() for key, value in gt_eval.items()}
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_current.pth'))
                    summary_fn(gt_eval, model_output_eval, writer, total_steps)
                    for arr in model_out_arrs:
                        del arr
                    del model_output_eval['model_in']
                    del model_output_eval['model_out']
                    torch.cuda.empty_cache()

                if not use_lbfgs:
                    optim.zero_grad()
                    train_loss.backward()

                    if clip_grad:
                        if isinstance(clip_grad, bool):
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                    optim.step()

                pbar.update(1)

                if not total_steps % steps_til_summary:
                    tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))

                    if val_dataloader is not None:
                        print("Running validation set...")
                        model.eval()
                        with torch.no_grad():
                            val_losses = []
                            for (model_input, gt) in val_dataloader:
                                model_output = model(model_input)
                                val_loss = loss_fn(model_output, gt)
                                val_losses.append(val_loss)

                            writer.add_scalar("val_loss", np.mean(val_losses), total_steps)
                        model.train()

                total_steps += 1

        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))


class LinearDecaySchedule():
    def __init__(self, start_val, final_val, num_steps):
        self.start_val = start_val
        self.final_val = final_val
        self.num_steps = num_steps

    def __call__(self, iter):
        return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)
