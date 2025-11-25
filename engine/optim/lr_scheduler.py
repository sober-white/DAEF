"""   
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.  
"""  

import math    
from functools import partial
import matplotlib.pyplot as plt     
from ..extre_module.utils import plt_settings, TryExcept     
   
def flat_cosine_schedule(total_iter, warmup_iter, flat_iter, no_aug_iter, current_iter, init_lr, min_lr):  
    """  
    Computes the learning rate using a warm-up, flat, and cosine decay schedule.
    计算基于 warm-up、flat 以及 cosine 衰减的学习率。
    Args:
        total_iter (int): Total number of iterations. 总迭代次数。  
        warmup_iter (int): Number of iterations for warm-up phase. 预热阶段的迭代次数。 
        flat_iter (int): Number of iterations for flat phase. 平坦阶段的迭代次数（warm-up 之后，cosine 衰减之前）。
        no_aug_iter (int): Number of iterations for no-augmentation phase. 无增强阶段的迭代次数（最后的学习率固定为 min_lr）。     
        current_iter (int): Current iteration. 当前迭代次数。
        init_lr (float): Initial learning rate. 初始学习率。   
        min_lr (float): Minimum learning rate. 最小学习率。   

    Returns: 
        float: Calculated learning rate.
    """
    # **1. 预热阶段（warm-up）**：使用平方增长策略，使学习率逐渐增加到 init_lr  
    if current_iter <= warmup_iter:
        return init_lr * (current_iter / float(warmup_iter)) ** 2
    # **2. 平坦阶段（flat）**：保持学习率恒定为 init_lr     
    elif warmup_iter < current_iter <= flat_iter:
        return init_lr   
    # **3. 无增强阶段（no-augmentation）**：保持学习率恒定为 min_lr
    elif current_iter >= total_iter - no_aug_iter: 
        return min_lr     
    # **4. 余弦衰减阶段（cosine decay）**：
    else: 
        # 计算余弦衰减因子  
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (current_iter - flat_iter) /    
                                           (total_iter - flat_iter - no_aug_iter)))   
        # 计算余弦衰减因子     
        return min_lr + (init_lr - min_lr) * cosine_decay

     
class FlatCosineLRScheduler:
    """ 
    Learning rate scheduler with warm-up, optional flat phase, and cosine decay following RTMDet.    
    具有 warm-up、flat 和 cosine 衰减的学习率调度器，类似于 RTMDet。     
    
    Args:
        optimizer (torch.optim.Optimizer): Optimizer instance. PyTorch 优化器实例。    
        lr_gamma (float): Scaling factor for the minimum learning rate. 最小学习率相对于初始学习率的缩放因子。 
        iter_per_epoch (int): Number of iterations per epoch. 每个 epoch 的迭代次数（batch 数量）。
        total_epochs (int): Total number of training epochs. 训练的总 epoch 数。 
        warmup_epochs (int): Number of warm-up epochs. 预热阶段的迭代次数。
        flat_epochs (int): Number of flat epochs (for flat-cosine scheduler). 平坦阶段的 epoch 数（平稳学习率）。 
        no_aug_epochs (int): Number of no-augmentation epochs. 无增强阶段的 epoch 数（学习率锁定为 min_lr）。
        scheduler_type (str): 学习率调度类型（默认为 "cosine"）。  
    """

    '''
    学习率变化过程    
    
    假设：
        •	init_lr = 0.01
        •	min_lr = 0.0001    
        •	total_iter = 10000
        •	warmup_iter = 1000   
        •	flat_iter = 3000
        •	no_aug_iter = 500
    
    则：
        1.	[0 - 1000] 预热阶段：学习率从 0 增长到 0.01（二次方增长）。
        2.	[1000 - 3000] 平坦阶段：学习率保持 0.01。
        3.	[3000 - 9500] 余弦衰减阶段：学习率从 0.01 逐渐降低到 0.0001。   
        4.	[9500 - 10000] 无增强阶段：学习率保持 0.0001。
    '''     

    def __init__(self, optimizer, lr_gamma, iter_per_epoch, total_epochs, 
                 warmup_iter, flat_epochs, no_aug_epochs, scheduler_type="cosine", lr_scyedule_save_path=None): 
        # **1. 计算基础学习率（initial_lr）和最小学习率（min_lr）**
        self.base_lrs = [group["initial_lr"] for group in optimizer.param_groups]  # 获取优化器中每组参数的初始学习率     
        self.min_lrs = [base_lr * lr_gamma for base_lr in self.base_lrs]  # 计算最小学习率（init_lr * lr_gamma）   

        # **2. 计算不同阶段的迭代次数**
        total_iter = int(iter_per_epoch * total_epochs)  # 总训练迭代次数 = 迭代数/epoch * 总 epoch 数  
        no_aug_iter = int(iter_per_epoch * no_aug_epochs)  # 无增强阶段的迭代次数
        flat_iter = int(iter_per_epoch * flat_epochs)  # 平坦阶段的迭代次数
        if flat_iter > total_iter:  
            flat_iter = total_iter - warmup_iter   

        # **3. 打印关键超参数信息**     
        print(self.base_lrs, self.min_lrs, total_iter, warmup_iter, flat_iter, no_aug_iter)   

        # **4. 绑定 `flat_cosine_schedule` 计算函数**
        self.lr_func = partial(flat_cosine_schedule, total_iter, warmup_iter, flat_iter, no_aug_iter)    

        for i, _ in enumerate(optimizer.param_groups):
            plot_lr_schedule(total_iter, warmup_iter, flat_iter, no_aug_iter, self.base_lrs[i], self.min_lrs[i], lr_scyedule_save_path / f"lr_schedule_{i}.png") 

    def step(self, current_iter, optimizer):
        """
        Updates the learning rate of the optimizer at the current iteration.   
 
        Args:
            current_iter (int): Current iteration.   
            optimizer (torch.optim.Optimizer): Optimizer instance.
        """   
        # 遍历优化器中的参数组，更新学习率    
        for i, group in enumerate(optimizer.param_groups): 
            group["lr"] = self.lr_func(current_iter, self.base_lrs[i], self.min_lrs[i]) # 计算并设置新学习率   
        return optimizer # 返回更新后的优化器
  
@TryExcept('WARNING ⚠️ plot_lr_schedule failed.')
@plt_settings()
def plot_lr_schedule(total_iter, warmup_iter, flat_iter, no_aug_iter, init_lr, min_lr, save_path):
    is_four_stage = True
    iters = list(range(total_iter))   
    if flat_iter == (total_iter - warmup_iter):   
        is_four_stage = False
    lrs = [flat_cosine_schedule(total_iter, warmup_iter, flat_iter, no_aug_iter, i, init_lr, min_lr) for i in iters] 
   
    plt.figure(figsize=(8, 5))
    plt.plot(iters, lrs, label='Learning Rate')    
    plt.axvline(x=warmup_iter, color='r', linestyle='--', label='Warmup End')     
    if is_four_stage:     
        plt.axvline(x=flat_iter, color='g', linestyle='--', label='Flat End')
        plt.axvline(x=total_iter - no_aug_iter, color='b', linestyle='--', label='No Aug Start')   
    else:  
        plt.axvline(x=flat_iter + warmup_iter, color='g', linestyle='--', label='Flat End')
    plt.xlabel('Iterations')  
    plt.ylabel('Learning Rate')  
    plt.title('Flat Cosine Learning Rate Schedule')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)   
    plt.close('all')     

if __name__ == '__main__':
    # 参数设置
    total_iter = 10000
    warmup_iter = 1000    
    flat_iter = 3000 
    no_aug_iter = 500
    init_lr = 0.01  
    min_lr = 0.0001    

    # 绘制学习率曲线
    plot_lr_schedule(total_iter, warmup_iter, flat_iter, no_aug_iter, init_lr, min_lr)    
