"""  
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.     
"""     

import copy, torch, thop, time, os 
from calflops import calculate_flops
from typing import Tuple

RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"

def stats(
    cfg,    
    input_shape: Tuple=(1, 3, 640, 640), module=None) -> Tuple[int, dict]:
    
    base_size = cfg.train_dataloader.collate_fn.base_size
    input_shape = (1, 3, base_size, base_size)    

    if module:
        model_for_info = copy.deepcopy(module)
    else: 
        model_for_info = copy.deepcopy(cfg.model).deploy() 

    try:   
        flops, macs, _ = calculate_flops(model=model_for_info,   
                                            input_shape=input_shape,     
                                            output_as_string=True,   
                                            output_precision=4,   
                                            print_detailed=False)
    except: 
        print(RED + "calculate_flops failed.. using thop instead.." + RESET)   
        p = next(model_for_info.parameters())
        macs = thop.profile(model_for_info, inputs=[torch.randn(size=input_shape, device=p.device)], verbose=False)[0]
        macs, flops = thop.clever_format([macs, macs * 2], format="%.3f") 
    params = sum(p.numel() for p in model_for_info.parameters())
    del model_for_info    

    return params, {"Model FLOPs:%s   MACs:%s   Params:%s" %(flops, macs, params)}    
    
def get_weight_size(module):     
    timestamp = f'{int(time.time() * 1000)}.pth' # 毫秒级时间戳
    torch.save(module, timestamp)
    stats = os.stat(timestamp)
    print(ORANGE + f'-------------------- only model size: {stats.st_size / (1024 ** 2):.1f} MB --------------------' + RESET)  
    os.remove(timestamp)     
