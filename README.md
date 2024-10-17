# DR-CoT
Dynamic Recursive Chain of Thought with Meta Reasoning for parameter efficient models.


## Training

Model is trained on the gpqa dataset, using its main and diamond subset.

## Results 

| Evaluation Method and Model               | Main Set (%) | Diamond Set (%) |
|-------------------------------------------|--------------|-----------------|
| Zero-Shot Llama-2-70B-chat                | 27.6         | 26.5            |
| Few-Shot Llama-2-70B-chat                 | 26.9         | 24.0            |
| Zero-Shot CoT Llama-2-70B-chat            | 28.5         | 31.1            |
| Few-Shot CoT Llama-2-70B-chat             | 29.1         | 28.1            |
| Zero-Shot GPT-3.5-turbo-16k               | 29.8         | 30.6            |
| Few-Shot GPT-3.5-turbo-16k                | 28.9         | 26.0            |
| Zero-Shot CoT GPT-3.5-turbo-16k           | 28.9         | 28.1            |
| Few-Shot CoT GPT-3.5-turbo-16k            | 28.0         | 29.6            |
| Zero-Shot GPT-4                           | 32.1         | 34.2            |
| Few-Shot GPT-4                            | 38.1         | 39.3            |
| Zero-Shot CoT GPT-4                       | 39.5         | 37.5            |
| Few-Shot CoT GPT-4                        | 39.7         | 38.8            |
| GPT-4 with search                         | 28.7         | 27.6            |
| **DR-CoT roberta-base(mine)**             | 26.7         | 0.30            | 
| **DR-CoT deberta-v3-base(mine)**          | -            | -               |
| **DR-CoT electra-base-generator(mine)**   | 27.7         | 0.325           |
| **DR-CoT bert-base-uncased(mine)**        | 26.6         | 0.35            |

## Citations
```
@inproceedings{rein2024gpqa,
      title={{GPQA}: A Graduate-Level Google-Proof Q\&A Benchmark},
      author={David Rein and Betty Li Hou and Asa Cooper Stickland and Jackson Petty and Richard Yuanzhe Pang and Julien Dirani and Julian Michael and Samuel R. Bowman},
      booktitle={First Conference on Language Modeling},
      year={2024},
      url={https://openreview.net/forum?id=Ti67584b98}
}
```
```
@article{DBLP:journals/corr/abs-1907-11692,
  author    = {Yinhan Liu and
               Myle Ott and
               Naman Goyal and
               Jingfei Du and
               Mandar Joshi and
               Danqi Chen and
               Omer Levy and
               Mike Lewis and
               Luke Zettlemoyer and
               Veselin Stoyanov},
  title     = {RoBERTa: {A} Robustly Optimized {BERT} Pretraining Approach},
  journal   = {CoRR},
  volume    = {abs/1907.11692},
  year      = {2019},
  url       = {http://arxiv.org/abs/1907.11692},
  archivePrefix = {arXiv},
  eprint    = {1907.11692},
  timestamp = {Thu, 01 Aug 2019 08:59:33 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1907-11692.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
```
@inproceedings{
he2021deberta,
title={DEBERTA: DECODING-ENHANCED BERT WITH DISENTANGLED ATTENTION},
author={Pengcheng He and Xiaodong Liu and Jianfeng Gao and Weizhu Chen},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=XPZIaotutsD}
}
```

```
@misc{clark2020electrapretrainingtextencoders,
      title={ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators}, 
      author={Kevin Clark and Minh-Thang Luong and Quoc V. Le and Christopher D. Manning},
      year={2020},
      eprint={2003.10555},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2003.10555}, 
}
```
```
@article{DBLP:journals/corr/abs-1810-04805,
  author    = {Jacob Devlin and
               Ming{-}Wei Chang and
               Kenton Lee and
               Kristina Toutanova},
  title     = {{BERT:} Pre-training of Deep Bidirectional Transformers for Language
               Understanding},
  journal   = {CoRR},
  volume    = {abs/1810.04805},
  year      = {2018},
  url       = {http://arxiv.org/abs/1810.04805},
  archivePrefix = {arXiv},
  eprint    = {1810.04805},
  timestamp = {Tue, 30 Oct 2018 20:39:56 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1810-04805.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
