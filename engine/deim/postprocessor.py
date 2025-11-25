"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)    
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""
  
import torch 
import torch.nn as nn  
import torch.nn.functional as F     
   
import torchvision

from ..core import register    
   
    
__all__ = ['PostProcessor']
     
    
def mod(a, b):  
    out = a - a // b * b  
    return out

 
@register()     
class PostProcessor(nn.Module):     
    # 共享的参数列表 
    __share__ = [
        'num_classes',
        'use_focal_loss',
        'num_top_queries', 
        'remap_mscoco_category'
    ]
  
    def __init__(   
        self,     
        num_classes=80,  # 目标类别数，默认 80（如 COCO 数据集）    
        use_focal_loss=True,  # 是否使用 Focal Loss
        num_top_queries=300,  # 选择前 num_top_queries 个目标
        remap_mscoco_category=False  # 是否对 COCO 数据集的类别进行重新映射
    ) -> None:   
        super().__init__()   
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries 
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category  
        self.deploy_mode = False  # 部署模式标志位，默认关闭

    def extra_repr(self) -> str:
        """
        额外的表示方法，在打印模型结构时显示     
        """
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'

    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        """  
        前向传播函数，处理检测头的输出，将其转换为最终检测结果。
        :param outputs: 预测的输出字典，包含 'pred_logits'（类别分数）和 'pred_boxes'（边界框）
        :param orig_target_sizes: 原始图像的尺寸，用于恢复边界框到原始图像尺度    
        :return: 处理后的检测结果，包含 labels、boxes、scores
        """    
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']

        # 将边界框从中心点宽高（cxcywh）格式转换为左上角右下角（xyxy）格式   
        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy') 
        # 根据原始图像尺寸缩放边界框
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1) 

        if self.use_focal_loss:  
            # 使用 Focal Loss 计算类别分数    
            scores = F.sigmoid(logits)
            # 选择最高的 num_top_queries 个目标
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            # 计算类别标签 
            labels = index % self.num_classes  # 取模计算类别索引     
            index = index // self.num_classes  # 计算 box 索引 
            # 根据索引选择对应的边界框     
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1])) 
        else:
            # 计算 softmax 归一化后的类别分数
            scores = F.softmax(logits, dim=-1)[:, :, :-1]  # 去掉背景类别 
            scores, labels = scores.max(dim=-1)  # 选择最大分数对应的类别   
            if scores.shape[1] > self.num_top_queries:     
                # 选择最高的 num_top_queries 个目标  
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1) 
                labels = torch.gather(labels, dim=1, index=index)  
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))   

        # 部署模式下，直接返回 labels、boxes、scores
        if self.deploy_mode: 
            return labels, boxes, scores
   
        # 如果需要映射 COCO 类别索引
        if self.remap_mscoco_category:
            from ..data.dataset import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)   
 
        # 组织输出结果    
        results = []
        for lab, box, sco in zip(labels, boxes, scores):   
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)

        return results

    def deploy(self):     
        """
        设置模型为部署模式（推理模式）     
        """ 
        self.eval()     
        self.deploy_mode = True
        return self