## Angular mc loss for deep metric learning

Tensorflow Implemenation of Deep Metric Learning with Angular Loss.
https://arxiv.org/abs/1708.01682

<img src="examples/fig_01.png" width="350">
<img src="examples/fig_02.png" width="350">

## NOTE:
+ All classes in an input batch should be different.
+ When you guys use this angular loss with N-pair loss, let ```with_l2_reg``` be ```False```. If not, let it be ```True```.
+ I implemented only an angular mc loss. triplet version will be updated.

## Reference
```
@inproceedings{wang2017deep,
  title={Deep metric learning with angular loss},
  author={Wang, Jian and Zhou, Feng and Wen, Shilei and Liu, Xiao and Lin, Yuanqing},
  booktitle={2017 IEEE International Conference on Computer Vision (ICCV)},
  pages={2612--2620},
  year={2017},
  organization={IEEE}
}
```
