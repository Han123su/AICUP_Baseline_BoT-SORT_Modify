# AICUP baseline
* ###本次README.md 參考自 README(original).md

	Reference: (https://github.com/ricky-696/AICUP_Baseline_BoT-SORT)

# Package Version
## 在環境的套件配置上，本次參賽使用的 package version 如下
python==3.9.7

torch==1.13.1

torchvision==0.14.1

tensorboard==2.16.2

scikit-image==0.22.0

scikit-learn==1.4.2

onnx==1.16.0

opencv-python==4.9.0.80

wheel==0.41.2

# REID: 提取特徵，不會做下游任務 (query、gallery)
## 在REID階段需動到的檔案路徑
bagtricks_R50-ibn.yml 超參數:
`<程式碼路徑>/AICUP_Baseline_BoT-SORT/fast_reid/configs/AICUP/`

Base-bagtricks.yml 被繼承的檔案:`<程式碼路徑>/AICUP_Baseline_BoT-SORT/fast_reid/configs/`

default.py 其呼叫的檔案:
`<程式碼路徑>/AICUP_Baseline_BoT-SORT/fast_reid/fastreid/config/`
---
* 跑training(用2顆GPU)
```shell
CUDA_VISIBLE_DEVICES='0,1' python3 fast_reid/tools/train_net.py --config-file fast_reid/configs/AICUP/bagtricks_R50-ibn.yml --num-gpus 4
```
* 跑完之後可以使用tensorboard視覺化，可open in browser
```shell
tensorboard --logdir=<程式碼路徑>/AICUP_Baseline_BoT-SORT/logs/<你的資料夾>/bagtricks_R50-ibn/
```
	
# YOLO v7 -> 物件偵測
AI_CUP.yml 檔案路徑:
`<程式碼路徑>/AICUP_Baseline_BoT-SORT/yolov7/data/`

改yolo架構 yolov7-AICUP.yaml 檔案路徑:
`<程式碼路徑>/AICUP_Baseline_BoT-SORT/yolov7/cfg/training/`

超參數 hyp.scratch.custom.yaml 檔案路徑:
`<程式碼路徑>/AICUP_Baseline_BoT-SORT/yolov7/data/`
	
* ### 跑 training (看用幾顆 GPU)

單張 GPU
```shell
python yolov7/train.py --device 0 --batch-size 8 --epochs 50 --data yolov7/data/AICUP.yaml --img 1280 1280 --cfg yolov7/cfg/training/yolov7-AICUP.yaml --weights 'pretrained/yolov7-e6e.pt' --name yolov7-AICUP --hyp data/hyp.scratch.custom.yaml
```
多張 GPU
```shell
python -m torch.distributed.launch --nproc_per_node 4 yolov7/train.py --device 0,1 --batch-size 8 --epochs 50 --data yolov7/data/AICUP.yaml --img 1280 1280 --cfg yolov7/cfg/training/yolov7-AICUP.yaml --weights 'pretrained/yolov7-e6e.pt' --name yolov7-AICUP --hyp data/hyp.scratch.custom.yaml
```
跑完之後用tensorboard可視化 open in browser
```shell
tensorboard --logdir=<程式碼路徑>/AICUP_Baseline_BoT-SORT/runs/train/yolov7-AICUP<你的資料夾>
```
	
# Bot-SORT Tracking
* ### mc_demo_yolov7.py 算演算法 (放YOLO的結果)
```shell
bash tools/track_all_timestamps.sh --weights runs/train/yolov7-AICUP/weights/best.pt --source-dir /data/NAS/ComputeServer/slicepaste/AI_CUP/train/images/<timestamp> --device 0 --fast-reid-config fast_reid/configs/AICUP/bagtricks_R50-ibn.yml --fast-reid-weights logs/<資料夾>/bagtricks_R50-ibn/model_00<第幾個>.pth
```
tracking結果路徑:
	`<程式碼路徑>/AICUP_Baseline_BoT-SORT/runs/detect/<timestamp>`

到此步驟便可產生 Bot-SORT Tracking 的 txt



# output評分
* ### 把 GT 轉換成 MOT15 格式
```shell
python tools/datasets/AICUP_to_MOT15.py --AICUP_dir <原始資料路徑> --MOT15_dir <欲儲存的資料路徑>
```

把 Tracking 的 txt 拉到 predict_result 資料夾

原檔 `<程式碼路徑>/AICUP_Baseline_BoT-SORT/runs/detect/<timestamp>/<XXX>.txt`

放到 `<程式碼路徑>/AICUP_Baseline_BoT-SORT/runs/predict_result/`

* ### 跑 evaluate
```shell
python tools/evaluate.py --gt_dir <輸出MOT15的路徑> --ts_dir runs/predict_result/
```


