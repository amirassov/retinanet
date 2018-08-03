# RetinaNet for Object Detection.

#### Example usage:
````
CUDA_VISIBLE_DEVICES="{gpus}" PYTHONPATH=path/fpnssd python run.py \
    --config=path/config.yml \
    --samples=path/samples.json \
    --folds=path/folds.csv \
    --fold=fold \
    --data_dir=path/data
````

#### TODO

- [ ] `cuda -> to_device`
- [ ] rewrite the predict method of ssd
- [ ] configurable fpn
- [ ] [common](https://github.com/Scitator/pytorch-common)


#### References: 
1. [TorchCV](https://github.com/kuangliu/torchcv)
2. [ChainerCV](https://github.com/chainer/chainercv)
3. [Albumentations](https://github.com/albu/albumentations)
