# X-VoE - eXplaining Violation of Expectation in Physical Events
[Bo Dai](https://daibopku.github.io/daibo/), Linge Wang, [Chi Zhang](https://wellyzhang.github.io/), [Baoxiong Jia](https://buzz-beater.github.io/), [Zeyu Zhang](https://zeyuzhang.com/), [Yixin Zhu](https://yzhu.io/), [Song-Chun Zhu](http://www.stat.ucla.edu/~sczhu/)

Code for our ArXiv paper "X-VoE: eXplaining Violation of Expectation in Physical Events": 

<img src="https://github.com/daibopku/XPL/blob/main/figure/explain.png" width="500">

## Abstract
Intuitive physics plays one of the most fundamental roles in helping people make sense of the physical world. Via intuitive physics, we engage in the physical activities in the world by predicting future events, making valid hypotheses of possible outcomes, and explicating observed events even from infancy. Yet, how to create artificial intelligence that learns and uses intuitive physics at the human level remains elusive for the learning community. In this work, we mitigate the gap by proposing a comprehensive benchmark X-VoE, evaluating intuitive physics learning via the Violation of Expectation (VoE) paradigm rooted in the community of developmental psychology. Compared to existing datasets, X-VoE focuses on the three aspects of explaining VoE in intuitive physics learning: predicting future states, making hypotheses of outcomes, and explicating observed events. Specifically, we design the three distinctive settings for each VoE scenario. Apart from measuring the performance of off-the-shelf models on our benchmark, we also propose an explanation-based learning system that jointly learns physics dynamics and infers future states by only observing visual sequences. In experiments, we demonstrate that our model is more consistent with human commonsense in all the scenarios in the dataset compared to others. Furthermore, our model can visually explain a VoE event by visualizing the internal representation. Finally, we discuss the implication of experimental results and future direction.

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
title = {X-VoE: eXplaining Violation of Expectation in Physical Events},
author = {Bo Dai, Linge Wang, Chi Zhang, Baoxiong Jia, Zeyu Zhang, Yixin Zhu, Song-Chun Zhu},
booktitle = {ArXiv},
year = {2023}
}
```
