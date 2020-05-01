# CentripetalNet

## env
```bash
conda create --name CentripetalNet -y python=3.6
source activate CentripetalNet

conda install pytorch=0.4.1 torchvision cuda92 -c pytorch

conda install pytorch=0.4.1 torchvision cuda90 -c pytorch

conda install cython -y
pip install matplotlib
conda install pillow
pip install opencv-python


pip install pytest-runner
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .

mkdir data
ln -s $COCO_ROOT data

```
## Run train
server 124
```bash
source activate CertripetalNet
python tools/train.py configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712.py --gpus 1 --work_dir work_dirs

```

## Run test 
```bash
python tools/test.py /home/syh/CentripetalNet/configs/centripetalnet_mask_hg104_det01.py /home/syh/CentripetalNet/mmdetection/work_dirs/epoch_2.pth --out test_result.pkl 
```

### Prepare COCO dataset.

It is recommended to symlink the dataset root to `$MMDETECTION/data`.

```
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
│   ├── VOCdevkit
│   │   ├── VOC2007
│   │   ├── VOC2012

```