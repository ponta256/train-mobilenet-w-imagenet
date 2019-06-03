Code for training MobileNet V1 with ImageNet data

Training (adjust it for your envrionment accordingly)
<pre>
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_basenet.py -a mobilenet /mnt/ssd/imagenet/ --batch-size=2048 --lr=0.001 --workers=10
</pre>

Evaluation
<pre>
python eval_basenet.py -a mobilenet --weight=model_best_weight.pth /mnt/ssd/imagenet/
</pre>
