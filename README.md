# BandControlNet: Fine-Grained Spatiotemporal Features for Steerable Popular Music Generation
## Introduction
This is the official implementation of BandControlNet
[![GitHub](https://img.shields.io/badge/GitHub-demo%20page-blue?logo=Github&style=flat-round)](https://chinglohsiu.github.io/files/bandcontrolnet.html)

## Demo
- [https://chinglohsiu.github.io/files/bandcontrolnet.html](https://chinglohsiu.github.io/files/bandcontrolnet.html)

<img alt="BandControlNet architecture" src="img/model_overall.png">

## Description
This paper presents BandControlNet, a controllable music cover generation model that:
1. Spatiotemporal features are proposed to offer fine-grained controls on the granularity of every bar and every track for complex multitrack music.
2. We design a novel music representation called REMI_Track which tokenizes the multitrack music into multiple separated token sequences and combines BPE techniques to further compress the sequences.
3. We adapt the parallel Transformers framework to accommodate the multiple sequences under REMI_Track representation and propose SE-SA and CTT modules to boost the structure and inter-track dependency modeling for multitrack music respectively.

If you have any questions or requests, please write to chinglohsiu[AT]gmail[DOT].com
