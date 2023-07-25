Multi-Symmetry Ensembles
---------------------------------------------------------------
<p align="center">
  <img width="500" alt="image" src="https://github.com/clott3/multi-sym-ensem/assets/55004415/25c944c9-2861-4cc6-ab72-0ec7808e58c4">
</p>



Official PyTorch implementation of [Multi-Symmetry Ensembles (MSE)](https://arxiv.org/abs/2303.02484).

```
@article{loh2023multisymmetry,
      title={Multi-Symmetry Ensembles: Improving Diversity and Generalization via Opposing Symmetries}, 
      author={Loh, Charlotte and Han, Seungwook and Sudalairaj, Shivchander and Dangovski, Rumen and Xu, Kai and Wenzel, Florian and Solja{\v{c}}i{\'c}, Marin and Srivastava, Akash},
      journal={arXiv preprint arXiv:2303.02484},
      year={2023}
}
```

## Reproducing results
### Pre-training Equivariant and Invariant models (for four-fold rotation)
Equivariant models:
```
python main.py --data <path-to-data> --checkpoint-dir <checkpoint-dir> --log-dir <tensorboard-log-dir> --rotate eq --lmbd 0.4 --exp <name-of-exp>
```
Invariant models:
```
python main.py --data <path-to-data> --checkpoint-dir <checkpoint-dir> --log-dir <tensorboard-log-dir> --rotate inv --exp <name-of-exp>
```
SimCLR baseline models:
```
python main.py --data <path-to-data> --checkpoint-dir <checkpoint-dir> --log-dir <tensorboard-log-dir> --exp <name-of-exp>
```
This part of the code is adapted from the [Equivariant-SSL repo](https://github.com/rdangovs/essl/edit/main/imagenet/simclr/).
### Fine-tuning
Equivariant and baseline models:
```
python eval_ensem.py --data <path-to-data> --eval-mode finetune --dataset imagenet --pretrained <path-to-pretrained-ckpt> --checkpoint-dir <checkpoint-dir> --log-dir <tensorboard-log-dir> --exp-id <name-of-exp>  
```
Invariant models:
```
python eval_ensem.py --data <path-to-data> --eval-mode finetune --dataset imagenet --pretrained <path-to-pretrained-ckpt> --checkpoint-dir <checkpoint-dir> --log-dir <tensorboard-log-dir> --exp-id <name-of-exp> --lr-classifier 0.004 --lr-backbone 0.004  
```
### Ensemble evaluation
Example of MSE with 2-member equivariant + invariant models:
```
python eval_ensem.py --data <path-to-data> --eval-mode freeze --dataset imagenet --pretrained <paths-to-finetuned-ckpts-separated-by-space> 
```

## Community

*[Let us know](mailto:cloh@mit.edu)
about interesting work with MSE and we will spread the word here.*

*Our work is accepted at ICML 2023.*

## License

This project is released under MIT License, which allows commercial use. See [LICENSE](LICENSE) for details.
