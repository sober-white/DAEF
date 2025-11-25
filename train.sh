set CUDA_VISIBLE_DEVICES=0
torchrun --master_port=7777 --nproc_per_node=1 train.py -c configs/deim_dfine/deim_hgnetv2_n_battery.yml --seed=0


# deim_dfine_hgnetv2_s_coco_120e.pth为coco预训练权重
python train.py -c configs/deim_dfine/deim_hgnetv2_n_battery.yml -d cuda:2 --seed=0 -t deim_dfine_hgnetv2_n_coco_160e.pth
python train.py -c configs/test/deim_hgnetv2_n_battery.yml --seed=0 -t deim_dfine_hgnetv2_n_coco_160e.pth
python train.py -c configs/deim_dfine/deim_hgnetv2_s_battery.yml -d cuda:2 --seed=0 -t deim_dfine_hgnetv2_s_coco_120e.pth
python train.py -c configs/deim_dfine/deim_hgnetv2_n_battery.yml -d cuda:0 --seed=0 -r deim_outputs/deim_hgnetv2_n_battery_hyperWT_2/last.pth
python train.py -c configs/deim_dfine/deim_hgnetv2_n_battery.yml --use-amp --seed=0

# best_stg2.pth为训练132轮最佳map50:95的精度权重
python train.py -c F:/cyc/DEIM-main/configs/deim_dfine/deim_hgnetv2_n_battery.yml --test-only -r F:/cyc/DEIM-main/deim_outputs/deim_hgnetv2_n_neu_base_200e/best_stg2.pth
python train.py -c configs/deim_dfine/deim_hgnetv2_s_battery.yml --test-only -r deim_outputs/deim_hgnetv2_s_battery_EAN_FAL/best_stg2.pth
# FLOPs,MACs,Params
python tools/benchmark/get_info.py -c configs/test/deim_hgnetv2_n_battery.yml

#推理
python tools/inference/torch_inf.py -c configs/deim_dfine/deim_hgnetv2_n_battery.yml -r deim_outputs/deim_hgnetv2_n_battery_EAN_FAL/best_stg1.pth --input F:/cyc/DEIM-main/datasets/battery-test/images --output inference_results/exp -t 0.2

# Latency
trtexec --onnx="deim_outputs/deim_hgnetv2_n_battery_EAN_FAL-ga/best_stg1.onnx" --saveEngine="model.engine" --fp16
python tools/deployment/export_onnx.py --check -c configs/deim_dfine/deim_hgnetv2_n_battery.yml -r deim_outputs/deim_hgnetv2_n_battery_EAN_FAL-ga/best_stg1.pth
python tools/benchmark/trt_benchmark.py --COCO_dir F:/cyc/DEIM-main/datasets/battery-test/images --engine_dir output/dfine_hgnetv2_n_battery

& E:/workspace/zhanghongchao/anaconda/anaconda3/envs/DEIM/python.exe

