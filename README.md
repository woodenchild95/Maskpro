# MaskPro：Linear-Space Probabilistic Learning for (N:M)-Sparsity 

## Usage

We implement the mask learning for large language models with [PyTorch](https://pytorch.org/). To ensure a fair comparison with prior approaches, we recommend adopting the following setup for side-by-side evaluation.

### Installation
```
git clone https://github.com/woodenchild95/Maskpro.git
pip install transformers==4.45.2 accelerate datasets SentencePiece protobuf
```

### Training
Before training, you need to pre-generate an initial mask. While random initialization is feasible, it often results in longer training cycles. Using a pre-initialized mask can help accelerate the training process. We recommend using [SparseGPT](https://github.com/IST-DASLab/sparsegpt.git)/[Wanda](https://github.com/locuslab/wanda.git) to generate a sparse model, and then running our `get_mask.py` script to extract and save the corresponding masks.

Our proposed refined PGE incorporates the loss values of the initial mask across different minibatches. These values can be computed during training by performing two forward passes with alternating masks. As a more convenient alternative, you can directly precompute and store the initial loss values:
```
python inference_loss.py --dataset_size 512 --batchsize 32
```
Then you can train masks for about 10,000 iterations:
```
python train.py --dataset_size 512 --batchsize 32 --lr 50 --logits 10 --epoch 625
```
You may freely adjust the dataset size and the batchsize. Please ensure that the settings remain consistent on both inference and training.

## Note
We have tested the method on both A100 and H100 GPUs. While A100 yields slightly lower performance compared to H100, likely due to precision differences at the hardware level, this effect is mostly noticeable for models larger than 7B. For such large models, we recommend using H100 or higher-end GPUs. For smaller models, the hardware difference has minimal impact.

## ToDo
- [ ] compress mask
- [ ] update pge files

## Citation
```
@article{sun2025maskpro,
  title={MaskPro: Linear-Space Probabilistic Learning for Strict (N: M)-Sparsity on Large Language Models},
  author={Sun, Yan and Zhang, Qixin and Yu, Zhiyuan and Zhang, Xikun and Shen, Li and Tao, Dacheng},
  journal={arXiv preprint arXiv:2506.12876},
  year={2025}
}
```
