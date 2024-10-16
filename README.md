# RecursiveCoT

The `recursive_cot` function generates reasoning iteratively, asking the model to predict an answer and assess its confidence at each step. If the model's confidence exceeds a set threshold, the process stops; otherwise, the reasoning is updated and fed back into the model as input for the next step. This continues until the model is confident enough or the maximum allowed steps are reached. Each step's prediction and reasoning are saved.

## Training

Model is trained on the gpqa dataset, using its main subset.

## Results 

| Evaluation Method and Model               | Main Set (%)           |
|-------------------------------------------|------------------------|
| Zero-Shot Llama-2-70B-chat                | 27.6                   |
| Few-Shot Llama-2-70B-chat                 | 26.9                   |
| Zero-Shot CoT Llama-2-70B-chat            | 28.5                   |
| Few-Shot CoT Llama-2-70B-chat             | 29.1                   |
| Zero-Shot GPT-3.5-turbo-16k               | 29.8                   |
| Few-Shot GPT-3.5-turbo-16k                | 28.9                   |
| Zero-Shot CoT GPT-3.5-turbo-16k           | 28.9                   |
| Few-Shot CoT GPT-3.5-turbo-16k            | 28.0                   |
| Zero-Shot GPT-4                           | 32.1                   |
| Few-Shot GPT-4                            | 38.1                   |
| Zero-Shot CoT GPT-4                       | 39.5                   |
| Few-Shot CoT GPT-4                        | 39.7                   |
| GPT-4 with search                         | 28.7                   |
| **ReCoT RoBERTa(mine)**                        | 23.3                   |

```
@inproceedings{rein2024gpqa,
      title={{GPQA}: A Graduate-Level Google-Proof Q\&A Benchmark},
      author={David Rein and Betty Li Hou and Asa Cooper Stickland and Jackson Petty and Richard Yuanzhe Pang and Julien Dirani and Julian Michael and Samuel R. Bowman},
      booktitle={First Conference on Language Modeling},
      year={2024},
      url={https://openreview.net/forum?id=Ti67584b98}
}
```
