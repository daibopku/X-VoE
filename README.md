# X-VoE - eXplaining Violation of Expectation in Physical Events
[Bo Dai](https://daibopku.github.io/daibo/), Linge Wang, [Chi Zhang](https://wellyzhang.github.io/), [Baoxiong Jia](https://buzz-beater.github.io/), [Zeyu Zhang](https://zeyuzhang.com/), [Yixin Zhu](https://yzhu.io/), [Song-Chun Zhu](http://www.stat.ucla.edu/~sczhu/)

Code for our ArXiv paper "X-VoE: Measuring eXplanatory Violation of Expectation in Physical Events": 

<img src="https://github.com/daibopku/XPL/blob/main/figure/explain.png" width="500">

## Abstract
Intuitive physics plays one of the most fundamental roles in helping people make sense of the physical world. Via intuitive physics, we understand physical activities in the world by predicting future events and explicating observed events, even from infancy. Yet, how to create artificial intelligence that learns and uses intuitive physics at the human level remains elusive for the learning community. In this work, we propose a comprehensive benchmark X-VoE both as an evaluative tool and a challenge for artificial agents, measuring their intuitive physics learning via the Violation of Expectation (VoE) paradigm rooted in the community of developmental psychology. Compared to existing datasets, X-VoE places higher demands on the explanatory ability of intuitive physics models. Specifically, in each VoE scenario, we design three distinctive settings to verify the models' understanding of the events, demanding not only simple predictive but also explicative abilities. Apart from measuring the performance of off-the-shelf models on our benchmark, we also devise an explanation-based learning system that jointly learns physics dynamics and infers occluded object states by learning only from observed visual sequences (without the unobserved occlusion label). In experiments, we demonstrate that our model shows more consistent behaviors aligned with human commonsense in X-VoE. Crucially, our model can visually explain a VoE event by reconstructing the hidden scenes. Finally, we discuss the implication of experimental results and future direction.

## Environment
The project is developed and tested with python 3.8, tensorflow 2.8 and cuda 11.5, but any version newer than that should work. For simple installation of the packages needed, please install requirements.txt.
```
pip install -r requirements.txt
```
## Dataset and Checkpoint

The dataset can be downloaded using the script download_dataset.sh.
```
bash scripts/download_dataset.sh
```
The dataset is saved as tfrecord file and can be read by video_read.py.

The checkpoint of perception, xpl and plato can be downloaded using the script download_pretrained_model.sh
```
bash scripts/download_pretrained_model.sh 
```

## Eval
The metrix in paper can be calculated by eval_xpl.py and eval_plato.py.
```
python -m eval.eval_xpl
python -m eval.eval_plato
```

## Visualize
The visualization of explaining result by xpl can be shown in visualize.ipynb

## Training
To train the perception model, simply execute:
```
python -m scripts.make_video_data
python -m train.train_perception
```

To train the xpl model, simply execute:
```
python -m train.train_xpl
```

To train the plato model, simply execute:
```
python -m train.train_plato
```

When using this code, please cite the paper:
```
@article{dai2023xvoe,
title = {X-VoE: Measuring eXplanatory Violation of Expectation in Physical Events},
author = {Bo Dai, Linge Wang, Chi Zhang, Baoxiong Jia, Zeyu Zhang, Yixin Zhu, Song-Chun Zhu},
booktitle = {ArXiv},
year = {2023}
}
```
