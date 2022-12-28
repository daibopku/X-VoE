# X-VoE - eXplaining Violation of Expectation in Physical Events
[Bo Dai](https://daibopku.github.io/daibo/),  [Yixin Zhu](https://yzhu.io/), 

Code for our ArXiv paper "X-VoE: eXplaining Violation of Expectation in Physical Events": 

<img src="https://github.com/daibopku/XPL/blob/main/figure/explain.png" width="500">

## Abstract
Intuitive physics plays one of the most fundamental roles in helping people make sense of the physical world. Via intuitive physics, we engage in the physical activities in the world by predicting future events, making valid hypotheses of possible outcomes, and explicating observed events even from infancy. Yet, how to create artificial intelligence that learns and uses intuitive physics at the human level remains elusive for the learning community. In this work, we mitigate the gap by proposing a comprehensive benchmark X-VoE, evaluating intuitive physics learning via the Violation of Expectation (VoE) paradigm rooted in the community of developmental psychology. Compared to existing datasets, X-VoE focuses on the three aspects of explaining VoE in intuitive physics learning: predicting future states, making hypotheses of outcomes, and explicating observed events. Specifically, we design the three distinctive settings for each VoE scenario. Apart from measuring the performance of off-the-shelf models on our benchmark, we also propose an explanation-based learning system that jointly learns physics dynamics and infers future states by only observing visual sequences. In experiments, we demonstrate that our model is more consistent with human commonsense in all the scenarios in the dataset compared to others. Furthermore, our model can visually explain a VoE event by visualizing the internal representation. Finally, we discuss the implication of experimental results and future direction.

## Code

In main.py, there is an example on how to run PhyDNet on the Moving MNIST dataset.

If you find this code useful for your research, please cite our paper:

```
@incollection{leguen20phydnet,
title = {X-VoE: eXplaining Violation of Expectation in Physical Events},
author = {Bo Dai, Linge Wang, Chi Zhang, Baoxiong Jia, Zeyu Zhang, Yixin Zhu, Song-Chun Zhu},
booktitle = {ArXiv},
year = {2023}
}
```
