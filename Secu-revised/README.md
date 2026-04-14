# SeCu
PyTorch Implementation for Our ICCV'23 Paper: "Stable Cluster Discrimination for Deep Clustering"

## Requirements
* Python 3.8
* PyTorch 1.6

## Training
Put the training dataset into ./data/train
```
rice revised:
python main.py .\data\train -j 8 -p 10 --lr 0.01 --epochs 101 --secu-num-ins 2023 --secu-alpha 240 --clr 0.001 --min-crop 0.2 --log secu-medoid --dist-url tcp://localhost:1234 --multiprocessing-distributed --world-size 1 --rank 0 --secu-tx 0.07 --use-medoid 1



```
- You can also modify the training parameters in main.py  
- secu-num-ins needs to be set to the dataset size (N)  
- secu-alpha needs to be set to 6 * N / 50  
- model will save at ./model

## Testing
Put the training dataset into ./data/test
```
rice revised:
python inference.py --model-path model/secu-medoid_0200.pth.tar --secu-num-ins 2023 --secu-alpha 240 --secu-tx 0.07

python inference_new_gt_txt.py --model-path model/best_model.pth.tar --secu-num-ins 40696 --secu-alpha 1000 --secu-tx 0.07 --data-name custom --backbone vit

```
- You can modify the clusters amount in config.py  

## Usage:
SeCu with size constraint for CIFAR-10
```
sh run_cifar10.sh 0
```

SeCu with entropy constraint for CIFAR-10
```
sh run_cifar10_entropy.sh 0
```

## Citation
If you use the package in your research, please cite our paper:
```
@inproceedings{qian2023secu,
  author    = {Qi Qian},
  title     = {Stable Cluster Discrimination for Deep Clustering},
  booktitle = {{IEEE/CVF} International Conference on Computer Vision, {ICCV} 2023},
  year      = {2023}
}
```



python main_org.py .\data -j 4 -p 10 --lr 0.01 --epochs 201 --secu-num-ins 16000 --secu-alpha 1000 --clr 0.001 --min-crop 0.2 --log secu --dist-url tcp://localhost:1234 --multiprocessing-distributed --world-size 1 --rank 0 --secu-tx 0.07 --use-medoid 1 --secu-lratio 0.7 --warm-up 30 -b 64 --backbone vit --secu-cst size-mml

python inference_new_gt_txt.py --model-path model/best_model.pth.tar --secu-num-ins 16000 --secu-alpha 1000 --secu-tx 0.07 --data-name custom --backbone vit


使用說明書
1.先到E:\han_clustering\Secu-revised\data資料夾內放入train/test 檔案(放入格式為.train/label_名稱/.npy or .jpg  檔案)
2.然後輸入訓練指令(有開好終端機了 ):
python main_org.py .\data -j 4 -p 10 --lr 0.01 --epochs 201 --secu-num-ins 16000 --secu-alpha 1000 --secu-k 8 9 10 --clr 0.001 --min-crop 0.2 --log secu --dist-url tcp://localhost:1234 --multiprocessing-distributed --world-size 1 --rank 0 --secu-tx 0.07 --use-medoid 1 --secu-lratio 0.7 --warm-up 30 -b 64 --backbone vit --secu-cst size-mml
其中有些參數需要根據資料集去修改如:
secu-num-ins 為資料總數
secu-alpha 常見設定為資料數*6/50(不超過每一類的類別數量)
secu-k 分群得類別數量(常見設定為類別數、+1、+2 三個數)
3.測試有兩步驟
3.1 到config.py 修改 clusters_amount = 分群數量(只能輸入訓練所輸入secu-k其中一個數字)
3.2 inference 參數(也是要根據以上訓練參數做修改):
python inference_new_gt_txt.py --model-path model/best_model.pth.tar --secu-num-ins 16000 --secu-alpha 1000 --secu-k 8 9 10 --secu-tx 0.07 --data-name custom --backbone vit