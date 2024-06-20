# @Time    : 2024/1/20 03:13
# @Author  : zhangchenming
import os
import shutil
import time
import glob
import torch
import torch.nn as nn
import torch.distributed as dist

from stereo.config.instantiate import instantiate
from stereo.utils import common_utils
from stereo.utils.common_utils import write_tensorboard
from stereo.evaluation import metric_names, metric_funcs


class Trainer:
    def __init__(self, args, cfg, logger, tb_writer):
        self.args = args
        self.cfg = cfg
        self.local_rank = args.local_rank
        self.global_rank = args.global_rank
        self.logger = logger
        self.tb_writer = tb_writer

        self.model = self.build_model()

        if self.args.run_mode in ['train', 'eval']:
            self.eval_set, self.eval_loader, self.eval_sampler = self.build_eval_loader()
        if self.args.run_mode == 'train':
            self.train_set, self.train_loader, self.train_sampler = self.build_train_loader()
            self.total_epochs = cfg.train_params.train_epochs
            self.last_epoch = -1
            # optimizer
            cfg.optimizer.params.model = self.model
            self.optimizer = instantiate(self.cfg.optimizer)
            # scheduler
            cfg.scheduler.optimizer = self.optimizer
            if 'total_steps' in self.cfg.scheduler:
                cfg.scheduler.total_steps = cfg.train_params.train_epochs * len(self.train_loader)
            self.scheduler = instantiate(cfg.scheduler)
            self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.train_params.mixed_precision)
            if cfg.train_params.resume_from_ckpt > -1:
                self.resume_ckpt()
            if 'clip_grad' in cfg:
                self.clip_gard = instantiate(cfg.clip_grad)

    def build_train_loader(self):
        self.cfg.train_loader.is_dist = self.args.dist_mode
        train_set, train_loader, train_sampler = instantiate(self.cfg.train_loader)
        self.logger.info('Total samples for train dataset: %d' % (len(train_set)))
        return train_set, train_loader, train_sampler

    def build_eval_loader(self):
        self.cfg.val_loader.is_dist = self.args.dist_mode
        eval_set, eval_loader, eval_sampler = instantiate(self.cfg.val_loader)
        self.logger.info('Total samples for eval dataset: %d' % (len(eval_set)))
        return eval_set, eval_loader, eval_sampler

    def build_model(self):
        model = instantiate(self.cfg.model)
        if self.cfg.train_params.freeze_bn:
            model = common_utils.freeze_bn(model)
            self.logger.info('Freeze the batch normalization layers')

        if self.cfg.train_params.use_sync_bn and self.args.dist_mode:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            self.logger.info('Convert batch norm to sync batch norm')
        model = model.to(self.local_rank)

        if self.args.dist_mode:
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[self.local_rank], output_device=self.local_rank,
                find_unused_parameters=self.cfg.train_params.find_unused_parameters)

        # load pretrained model
        pretrained_model = self.cfg.train_params.pretrained_model
        if pretrained_model:
            self.logger.info('Loading parameters from checkpoint %s' % pretrained_model)
            if not os.path.isfile(pretrained_model):
                raise FileNotFoundError
            common_utils.load_params_from_file(
                model, pretrained_model, device='cuda:%d' % self.local_rank,
                dist_mode=self.args.dist_mode, logger=self.logger, strict=False)
        return model

    def resume_ckpt(self):
        self.logger.info('Resume from ckpt:%d' % self.cfgs.MODEL.CKPT)
        ckpt_path = str(os.path.join(self.args.ckpt_dir, 'checkpoint_epoch_%d.pth' % self.cfgs.MODEL.CKPT))
        checkpoint = torch.load(ckpt_path, map_location='cuda:%d' % self.local_rank)
        self.last_epoch = checkpoint['epoch']
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scaler.load_state_dict(checkpoint['scaler_state'])
        if self.args.dist_mode:
            self.model.module.load_state_dict(checkpoint['model_state'])
        else:
            self.model.load_state_dict(checkpoint['model_state'])

    def train(self, current_epoch, tbar):
        self.model.train()
        if self.cfg.train_params.freeze_bn:
            self.model = common_utils.freeze_bn(self.model)
        if self.args.dist_mode:
            self.train_sampler.set_epoch(current_epoch)
        self.train_one_epoch(current_epoch=current_epoch, tbar=tbar)
        if self.args.dist_mode:
            dist.barrier()

    def evaluate(self, current_epoch):
        self.model.eval()
        self.eval_one_epoch(current_epoch=current_epoch)
        if self.args.dist_mode:
            dist.barrier()

    def save_ckpt(self, current_epoch):
        if self.global_rank == 0:
            ckpt_list = glob.glob(os.path.join(self.args.ckpt_dir, 'epoch_*'))
            ckpt_list.sort(key=os.path.getmtime)
            if len(ckpt_list) >= self.cfg.train_params.max_ckpt_save_num:
                for cur_file_idx in range(0, len(ckpt_list) - self.cfg.train_params.max_ckpt_save_num + 1):
                    shutil.rmtree(ckpt_list[cur_file_idx])

            output_dir = os.path.join(self.args.ckpt_dir, 'epoch_%d' % current_epoch)
            os.makedirs(output_dir, exist_ok=True)
            common_utils.save_checkpoint(self.model, self.optimizer, self.scheduler, self.scaler,
                                         self.args.dist_mode, output_dir)
        if self.args.dist_mode:
            dist.barrier()

    def train_one_epoch(self, current_epoch, tbar):
        start_epoch = self.last_epoch + 1
        total_loss = 0.0
        loss_func = self.model.module.get_loss if self.args.dist_mode else self.model.get_loss

        train_loader_iter = iter(self.train_loader)
        for i in range(0, len(self.train_loader)):
            self.optimizer.zero_grad()
            lr = self.optimizer.param_groups[0]['lr']

            start_timer = time.time()
            data = next(train_loader_iter)
            for k, v in data.items():
                data[k] = v.to(self.local_rank) if torch.is_tensor(v) else v
            data_timer = time.time()

            with torch.cuda.amp.autocast(enabled=self.cfg.train_params.mixed_precision):
                model_pred = self.model(data)
                infer_timer = time.time()
                loss, tb_info = loss_func(model_pred, data)

            # 不要在autocast下调用, calls backward() on scaled loss to create scaled gradients.
            self.scaler.scale(loss).backward()
            # 做梯度剪裁的时候需要先unscale, unscales the gradients of optimizer's assigned params in-place
            self.scaler.unscale_(self.optimizer)
            # 梯度剪裁
            if self.clip_gard is not None:
                self.clip_gard(self.model)
            # optimizer's gradients are already unscaled, so scaler.step does not unscale them
            self.scaler.step(self.optimizer)
            # Updates the scale for next iteration.
            self.scaler.update()

            self.scheduler.step()

            total_loss += loss.item()
            total_iter = current_epoch * len(self.train_loader) + i
            trained_time_past_all = tbar.format_dict['elapsed']
            single_iter_second = trained_time_past_all / (total_iter + 1 - start_epoch * len(self.train_loader))
            remaining_second_all = single_iter_second * (self.total_epochs * len(self.train_loader) - total_iter - 1)
            if total_iter % self.cfg.train_params.log_period == 0:
                message = ('Training Epoch:{:>2d}/{} Iter:{:>4d}/{} '
                           'Loss:{:#.6g}({:#.6g}) LR:{:.4e} '
                           'DataTime:{:.2f} InferTime:{:.2f}ms '
                           'Time cost: {}/{}'
                           ).format(current_epoch, self.total_epochs, i, len(self.train_loader),
                                    loss.item(), total_loss / (i + 1), lr,
                                    data_timer - start_timer, (infer_timer - data_timer) * 1000,
                                    tbar.format_interval(trained_time_past_all),
                                    tbar.format_interval(remaining_second_all))
                self.logger.info(message)

            tb_info.update({'scalar/train/lr': lr})
            if total_iter % self.cfg.train_params.log_period == 0 and self.local_rank == 0 and self.tb_writer is not None:
                write_tensorboard(self.tb_writer, tb_info, total_iter)

    @torch.no_grad()
    def eval_one_epoch(self, current_epoch):
        local_rank = self.local_rank

        epoch_metrics = {}
        for each in metric_names:
            epoch_metrics[each] = {'indexes': [], 'values': []}

        for i, data in enumerate(self.eval_loader):
            for k, v in data.items():
                data[k] = v.to(local_rank) if torch.is_tensor(v) else v

            with torch.cuda.amp.autocast(enabled=self.cfg.train_params.mixed_precision):
                infer_start = time.time()
                model_pred = self.model(data)
                infer_time = time.time() - infer_start

            disp_pred = model_pred['disp_pred']
            disp_gt = data["disp"]
            mask = (disp_gt < 192) & (disp_gt > 0.5)
            if 'occ_mask' in data:
                mask = mask & (data['occ_mask'] == 255.0)

            for each in metric_names:
                metric_func = metric_funcs[each]
                res = metric_func(disp_pred.squeeze(1), disp_gt, mask)
                epoch_metrics[each]['indexes'].extend(data['index'].tolist())
                epoch_metrics[each]['values'].extend(res.tolist())

            if i % self.cfg.train_params.log_period == 0:
                message = ('Evaluating Epoch:{:>2d} Iter:{:>4d}/{} InferTime: {:.2f}ms'
                           ).format(current_epoch, i, len(self.eval_loader), infer_time * 1000)
                self.logger.info(message)

        # gather from all gpus
        if self.args.dist_mode:
            dist.barrier()
            self.logger.info("Start reduce metrics.")
            for k in epoch_metrics.keys():
                indexes = torch.tensor(epoch_metrics[k]["indexes"]).to(local_rank)
                values = torch.tensor(epoch_metrics[k]["values"]).to(local_rank)
                gathered_indexes = [torch.zeros_like(indexes) for _ in range(dist.get_world_size())]
                gathered_values = [torch.zeros_like(values) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_indexes, indexes)
                dist.all_gather(gathered_values, values)
                unique_dict = {}
                for key, value in zip(torch.cat(gathered_indexes, dim=0).tolist(),
                                      torch.cat(gathered_values, dim=0).tolist()):
                    if key not in unique_dict:
                        unique_dict[key] = value
                epoch_metrics[k]["indexes"] = list(unique_dict.keys())
                epoch_metrics[k]["values"] = list(unique_dict.values())

        results = {}
        for k in epoch_metrics.keys():
            results[k] = torch.tensor(epoch_metrics[k]["values"]).mean()

        if local_rank == 0 and self.tb_writer is not None:
            tb_info = {}
            for k, v in results.items():
                tb_info[f'scalar/val/{k}'] = v.item()

            write_tensorboard(self.tb_writer, tb_info, current_epoch)

        self.logger.info(f"Epoch {current_epoch} metrics: {results}")
