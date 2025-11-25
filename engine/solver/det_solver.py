"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.     
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)     
Copyright (c) 2024 D-FINE authors. All Rights Reserved. 
"""  

import time
import json    
import datetime     

import torch

from ..misc import dist_utils, stats, get_weight_size
     
from ._solver import BaseSolver  
from .det_engine import train_one_epoch, evaluate
from ..optim.lr_scheduler import FlatCosineLRScheduler

RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m" 
coco_name_list = ['ap', 'ap50', 'ap75', 'aps', 'apm', 'apl', 'ar', 'ar50', 'ar75', 'ars', 'arm', 'arl']
 
class DetSolver(BaseSolver):     
   
    def fit(self, cfg_str):
        self.train()
        args = self.cfg    

        with open(self.output_dir / 'args.json', 'w') as json_file:     
            json_file.write(cfg_str)   
  
        # 计算模型参数量、FLOPs 等统计信息    
        n_parameters, model_stats = stats(self.cfg)    
        print(model_stats)     
        print("-"*42 + "Start training" + "-"*43)  
   
        # 初始化学习率调度器     
        self.self_lr_scheduler = False  
        if args.lrsheduler is not None:   
            iter_per_epoch = len(self.train_dataloader) 
            print("     ## Using Self-defined Scheduler-{} ## ".format(args.lrsheduler))
            self.lr_scheduler = FlatCosineLRScheduler(self.optimizer, args.lr_gamma, iter_per_epoch, total_epochs=args.epoches, 
                                                warmup_iter=args.warmup_iter, flat_epochs=args.flat_epoch, no_aug_epochs=args.no_aug_epoch, lr_scyedule_save_path=self.output_dir)    
            self.self_lr_scheduler = True
        # 统计需要训练的参数数量
        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad]) 
        print(f'number of trainable parameters: {n_parameters}')   
  
        top1 = 0     
        best_stat = {'epoch': -1, } 
        # evaluate again before resume training    
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model     
            test_stats, coco_evaluator = evaluate(   
                module,   
                self.criterion,
                self.postprocessor,  
                self.val_dataloader,   
                self.evaluator,
                self.device
            )
            for k in test_stats:   
                best_stat['epoch'] = self.last_epoch
                best_stat[k] = test_stats[k][0]   
                top1 = test_stats[k][0]    
                print(f'best_stat: {best_stat}')  

        best_stat_print = best_stat.copy()
        start_time = time.time()  
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epoches):
 
            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized(): 
                self.train_dataloader.sampler.set_epoch(epoch)    
    
            if epoch == self.train_dataloader.collate_fn.stop_epoch:  
                self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))    
                self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay    
                print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

            # 训练一个 epoch
            train_stats = train_one_epoch(
                self.self_lr_scheduler,
                self.lr_scheduler, 
                self.model, 
                self.criterion, 
                self.train_dataloader, 
                self.optimizer,  
                self.device,     
                epoch, 
                max_norm=args.clip_max_norm, 
                print_freq=args.print_freq, 
                ema=self.ema, 
                scaler=self.scaler, 
                lr_warmup_scheduler=self.lr_warmup_scheduler,     
                writer=self.writer,
                plot_train_batch_freq=args.plot_train_batch_freq,
                output_dir=self.output_dir,  
                epoches=args.epoches, # 总的训练次数  
                verbose_type=args.verbose_type    
            )
    
            if not self.self_lr_scheduler:  # update by epoch   
                if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():     
                    self.lr_scheduler.step()   
    
            self.last_epoch += 1
    
            if self.output_dir and epoch < self.train_dataloader.collate_fn.stop_epoch:
                checkpoint_paths = [self.output_dir / 'last.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:  
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths: 
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            # 训练一个epoch后计算模型指标
            module = self.ema.module if self.ema else self.model  
            test_stats, coco_evaluator = evaluate(     
                module,    
                self.criterion,
                self.postprocessor,
                self.val_dataloader, 
                self.evaluator,
                self.device
            ) 

            # TODO  
            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f'Test/{k}_{coco_name_list[i]}', v, epoch)    
   
                if k in best_stat:    
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])    
                else:
                    best_stat['epoch'] = epoch   
                    best_stat[k] = test_stats[k][0]
    
                if best_stat[k] > top1:
                    best_stat_print['epoch'] = epoch   
                    top1 = best_stat[k]  
                    if self.output_dir:   
                        if epoch >= self.train_dataloader.collate_fn.stop_epoch: # collate_fn.stop_epoch 代表 在多少epoch开始暂停数据增强
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg2.pth') # stg2 可以代表为无数据增强阶段
                        else: 
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg1.pth') # stg1 可以代表为有数据增强阶段     
     
                best_stat_print[k] = max(best_stat[k], top1)    
                print(f'best_stat: {best_stat_print}')  # global best    

                if best_stat['epoch'] == epoch and self.output_dir:    
                    if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        if test_stats[k][0] > top1: 
                            top1 = test_stats[k][0] 
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg2.pth')
                    else:
                        top1 = max(test_stats[k][0], top1) 
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg1.pth')     

                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    best_stat = {'epoch': -1, }     
                    self.ema.decay -= 0.0001 # 衰减因子 d 变小意味着当前模型参数在 EMA 更新中的占比更大    
                    self.load_resume_state(str(self.output_dir / 'best_stg1.pth')) # 这个代表是在stg2开始的时候会载入在stg1精度最高点模型来进行stg2的训练    
                    print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}') 
  
     
            log_stats = {  
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,  
                'n_parameters': n_parameters 
            }     
   
            if self.output_dir and dist_utils.is_main_process():  
                with (self.output_dir / "log.txt").open("a") as f:     
                    f.write(json.dumps(log_stats) + "\n")     
   
                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)   
                    if "bbox" in coco_evaluator.coco_eval:    
                        filenames = ['latest.pth']  
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')   
                        for name in filenames:     
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)  
 
        total_time = time.time() - start_time   
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))   
        print('Training time {}'.format(total_time_str)) 
     
 
    def val(self, ):   
        self.eval()
     
        module = self.ema.module if self.ema else self.model   
        module.deploy()  
        _, model_info = stats(self.cfg, module=module)
        print(ORANGE, "--------------------Model Info(fused)", model_info, "--------------------", RESET)
        get_weight_size(module)  
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,  
                self.val_dataloader, self.evaluator, self.device, True, self.output_dir)

        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
  
        return
