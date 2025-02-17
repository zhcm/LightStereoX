# @Time    : 2024/1/20 03:13
# @Author  : zhangchenming
import os
import shutil
import time
import glob
import math
import torch
import torch.nn as nn
import torch.distributed as dist

from stereo.config.instantiate import instantiate
from stereo.utils import common_utils
from stereo.evaluation import metric_names, metric_funcs


class Trainer:
    def __init__(self, args, cfg, logger, tb_writer):
        self.args = args
        self.cfg = cfg
        self.logger = logger
        self.tb_writer = tb_writer
        self.local_rank = args.local_rank
        self.global_rank = args.global_rank
        self.last_epoch = 0
        self.clip_gard = None
        self.warmup = None

        # model
        self.model = self.build_model()

        if self.args.run_mode in ['train', 'eval']:
            # val loader
            if 'is_dist' not in cfg.val_loader or cfg.val_loader.is_dist is None:
                cfg.val_loader.is_dist = args.dist_mode
            self.val_set, self.val_loader, self.val_sampler = instantiate(cfg.val_loader)
            self.logger.info('Total samples for val dataset: %d' % (len(self.val_set)))

        if self.args.run_mode == 'train':
            # train loader
            if 'is_dist' not in cfg.train_loader or cfg.train_loader.is_dist is None:
                cfg.train_loader.is_dist = args.dist_mode
            self.train_set, self.train_loader, self.train_sampler = instantiate(cfg.train_loader)
            self.logger.info('Total samples for train dataset: %d' % (len(self.train_set)))
            self.logger.info('Length of train loader: %d' % (len(self.train_loader)))
            if 'max_iter' in cfg.runtime_params and cfg.runtime_params.max_iter > 0:
                cfg.runtime_params.train_epochs = math.ceil(cfg.runtime_params.max_iter / len(self.train_loader))

            # optimizer
            cfg.optimizer.params.model = self.model
            self.optimizer = instantiate(self.cfg.optimizer)

            # scheduler
            cfg.scheduler.optimizer = self.optimizer
            if 'total_steps' in self.cfg.scheduler and cfg.scheduler.total_steps == -1:
                if 'max_iter' in cfg.runtime_params:
                    cfg.scheduler.total_steps = cfg.runtime_params.max_iter
                else:
                    cfg.scheduler.total_steps = cfg.runtime_params.train_epochs * len(self.train_loader)
            self.scheduler = instantiate(cfg.scheduler)

            # scaler
            self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.runtime_params.mixed_precision)

            # resume
            if cfg.runtime_params.resume_from_ckpt > -1:
                self.resume_ckpt()

            # warmup
            if 'warmup' in cfg:
                cfg.warmup.optimizer = self.optimizer
                cfg.warmup.last_step = self.last_epoch * len(self.train_loader) - 1
                self.warmup = instantiate(cfg.warmup)

            # clip grad
            if 'clip_grad' in cfg:
                self.clip_gard = instantiate(cfg.clip_grad)

    def build_model(self):
        model = instantiate(self.cfg.model)
        model = model.to(self.local_rank)
        # load pretrained model
        pretrained_model = self.cfg.runtime_params.pretrained_model
        if pretrained_model:
            self.logger.info('Loading parameters from checkpoint %s' % pretrained_model)
            if not os.path.isfile(pretrained_model):
                raise FileNotFoundError
            common_utils.load_params_from_file(model, pretrained_model, device='cpu', logger=self.logger, strict=False)
        # freeze bn
        if self.cfg.runtime_params.freeze_bn:
            model = common_utils.freeze_bn(model)
            self.logger.info('Freeze the batch normalization layers')
        # syncbn
        if self.cfg.runtime_params.use_sync_bn and self.args.dist_mode:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            self.logger.info('Convert batch norm to sync batch norm')
        # ddp
        if self.args.dist_mode:
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[self.local_rank], output_device=self.local_rank,
                find_unused_parameters=self.cfg.runtime_params.find_unused_parameters)
            self.logger.info('Convert model to DistributedDataParallel')
        return model

    def resume_ckpt(self):
        self.logger.info('Resume from ckpt:%d' % self.cfg.runtime_params.resume_from_ckpt)
        self.last_epoch = self.cfg.runtime_params.resume_from_ckpt
        checkpoint_dir = str(os.path.join(self.args.ckpt_dir, 'epoch_%d' % self.cfg.runtime_params.resume_from_ckpt))

        model_state = torch.load(os.path.join(checkpoint_dir, 'pytorch_model.bin'),
                                 map_location='cuda:%d' % self.local_rank)
        optim_state = torch.load(os.path.join(checkpoint_dir, 'optimizer.bin'),
                                 map_location='cuda:%d' % self.local_rank)
        scheduler_state = torch.load(os.path.join(checkpoint_dir, 'scheduler.bin'),
                                     map_location='cuda:%d' % self.local_rank)
        scaler_state = torch.load(os.path.join(checkpoint_dir, 'scaler.pt'),
                                  map_location='cuda:%d' % self.local_rank)

        self.scheduler.load_state_dict(scheduler_state)
        self.optimizer.load_state_dict(optim_state)
        self.scaler.load_state_dict(scaler_state)
        if self.args.dist_mode:
            self.model.module.load_state_dict(model_state)
        else:
            self.model.load_state_dict(model_state)

    def train(self, current_epoch, tbar):
        self.model.train()
        if self.cfg.runtime_params.freeze_bn:
            self.model = common_utils.freeze_bn(self.model)
        if self.args.dist_mode:
            self.train_sampler.set_epoch(current_epoch)
        self.train_one_epoch(current_epoch=current_epoch, tbar=tbar)
        if 'total_steps' not in self.cfg.scheduler:
            self.scheduler.step()
            if self.warmup:
                self.warmup.lrs = [group['lr'] for group in self.optimizer.param_groups]

        if self.args.dist_mode:
            dist.barrier()

    def evaluate(self, current_epoch):
        self.model.eval()
        self.eval_one_epoch(current_epoch=current_epoch)
        if self.args.dist_mode:
            dist.barrier()

    def save_ckpt(self, current_epoch):
        if self.global_rank == 0:
            # remove
            ckpt_list = glob.glob(os.path.join(self.args.ckpt_dir, 'epoch_*'))
            ckpt_list.sort(key=os.path.getmtime)
            if len(ckpt_list) >= self.cfg.runtime_params.max_ckpt_save_num:
                for cur_file_idx in range(0, len(ckpt_list) - self.cfg.runtime_params.max_ckpt_save_num + 1):
                    shutil.rmtree(ckpt_list[cur_file_idx])
            # save
            output_dir = os.path.join(self.args.ckpt_dir, 'epoch_%d' % current_epoch)
            os.makedirs(output_dir, exist_ok=True)
            common_utils.save_checkpoint(self.model, self.optimizer, self.scheduler, self.scaler,
                                         self.args.dist_mode, output_dir)
        if self.args.dist_mode:
            dist.barrier()

    def train_one_epoch(self, current_epoch, tbar):
        total_epochs = self.cfg.runtime_params.train_epochs
        total_loss = 0.0
        loss_func = self.model.module.get_loss if self.args.dist_mode else self.model.get_loss

        train_loader_iter = iter(self.train_loader)
        for i in range(0, len(self.train_loader)):

            # if i % 2000 == 0 and i != 0:
            #     self.save_ckpt(current_epoch=i)
            current_iter = (current_epoch - 1) * len(self.train_loader) + i
            if current_iter >= self.cfg.runtime_params.get('max_iter', 1e10):
                self.logger.info('Max iter reached.')
                break

            # zero grad
            self.optimizer.zero_grad()
            lr = self.optimizer.param_groups[0]['lr']

            start_timer = time.time()
            data = next(train_loader_iter)
            for k, v in data.items():
                data[k] = v.to(self.local_rank) if torch.is_tensor(v) else v
            data_timer = time.time()

            with torch.cuda.amp.autocast(enabled=self.cfg.runtime_params.mixed_precision):
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
            # scheduler
            if 'total_steps' in self.cfg.scheduler:
                self.scheduler.step()
            # warmup
            if self.warmup:
                with self.warmup.dampening():
                    pass

            # logging
            total_loss += loss.item()
            total_iter = (current_epoch - 1) * len(self.train_loader) + i
            trained_time_past_all = tbar.format_dict['elapsed']
            single_iter_second = trained_time_past_all / (total_iter + 1 - self.last_epoch * len(self.train_loader))
            remaining_second_all = single_iter_second * (total_epochs * len(self.train_loader) - total_iter - 1)
            if total_iter % self.cfg.runtime_params.log_period == 0:
                loss_message = ''
                for k, v in tb_info.items():
                    item_name = k.split('/')[-1]
                    if 'loss' not in item_name:
                        continue
                    loss_message += '{}:{:#.6g} '.format(item_name, v)

                message = ('Training Epoch:{:>2d}/{} Iter:{:>4d}/{} '
                           'Loss:{:#.6g}({:#.6g}) LR:{:.4e} '
                           'DataTime:{:.2f}ms InferTime:{:.2f}ms '
                           'Time cost: {}/{}'
                           ).format(current_epoch, total_epochs, i, len(self.train_loader),
                                    loss.item(), total_loss / (i + 1), lr,
                                    (data_timer - start_timer) * 1000, (infer_timer - data_timer) * 1000,
                                    tbar.format_interval(trained_time_past_all),
                                    tbar.format_interval(remaining_second_all))
                self.logger.info(message + '  ' + loss_message)

                tb_info.update({'scalar/train/lr': lr})
                if self.global_rank == 0 and self.tb_writer is not None:
                    common_utils.write_tensorboard(self.tb_writer, tb_info, total_iter)

    @torch.no_grad()
    def eval_one_epoch(self, current_epoch):
        all_indexes = []
        all_metrics = {}
        for each in metric_names:
            all_metrics[each] = []

        for i, data in enumerate(self.val_loader):
            for k, v in data.items():
                data[k] = v.to(self.local_rank) if torch.is_tensor(v) else v

            with torch.cuda.amp.autocast(enabled=self.cfg.runtime_params.mixed_precision):
                infer_start = time.time()
                model_pred = self.model(data)
                infer_time = time.time() - infer_start

            disp_pred = model_pred['disp_pred']
            disp_gt = data["disp"]
            mask = (disp_gt < self.cfg.runtime_params.eval_max_disp) & (disp_gt > 0.5)
            if 'occ_mask' in data:
                mask = mask & ~data['occ_mask']

            all_indexes.extend(data['index'].tolist())
            for each in metric_names:
                metric_func = metric_funcs[each]
                res = metric_func(disp_pred.squeeze(1), disp_gt, mask)
                all_metrics[each].extend(res.tolist())

            if i % self.cfg.runtime_params.log_period == 0:
                message = ('Evaluating Epoch:{:>2d} Iter:{:>4d}/{} InferTime: {:.2f}ms'
                           ).format(current_epoch, i, len(self.val_loader), infer_time * 1000)
                self.logger.info(message)

        # gather from all gpus
        if self.args.dist_mode:
            dist.barrier()
            self.logger.info("Start reduce metrics.")
            # gather index
            indexes = torch.tensor(all_indexes).to(self.local_rank)
            gathered_indexes = [torch.zeros_like(indexes) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_indexes, indexes)
            indexes = torch.cat(gathered_indexes)
            # gather metrics
            for each in metric_names:
                metric = torch.tensor(all_metrics[each]).to(self.local_rank)
                gathered_metric = [torch.zeros_like(metric) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_metric, metric)
                metric = torch.cat(gathered_metric)

                unique_dict = {}
                for key, value in zip(indexes.tolist(), metric.tolist()):
                    if key not in unique_dict:
                        unique_dict[key] = value
                all_metrics[each] = list(unique_dict.values())

        results = {}
        for each in metric_names:
            results[each] = round(torch.tensor(all_metrics[each]).mean().item(), 2)

        if self.global_rank == 0 and self.tb_writer is not None:
            tb_info = {}
            for k, v in results.items():
                tb_info[f'scalar/val/{k}'] = v
            common_utils.write_tensorboard(self.tb_writer, tb_info, current_epoch)

        self.logger.info(f"Epoch {current_epoch} metrics: {results}")
