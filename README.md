# End-to-End Beam Retrieval for Multi-Hop Question Answering
This is the repository for our paper "[End-to-End Beam Retrieval for Multi-Hop Question Answering](https://arxiv.org/abs/2308.08973)".

Our repository is under construction, feel free to contact us if you have any questions.

Cheers! Our paper has been accepted to NAACL 2024 main conference. And our results have been published on [MuSiQue-Ans](https://leaderboard.allenai.org/musique_ans/submissions/public) , [2WikiMultihopQA](https://github.com/Alab-NII/2wikimultihop) and [HotpotQA](https://hotpotqa.github.io/). 

## Download Data and Model
We use three original datasets [MuSiQue-Ans](https://github.com/StonyBrookNLP/musique/), [HotpotQA](https://hotpotqa.github.io/) and [2WikiMultihopQA](https://github.com/Alab-NII/2wikimultihop) for our main 
experiments and three paritial datasets sampled by [IRCoT](https://github.com/StonyBrookNLP/ircot).

We use [DeBERTa](https://huggingface.co/microsoft/deberta-v3-base) as our backbone model.

## Beam Retrieval
The code for our Beam Retrieval is in directory `retrieval`. To train our Beam Retrieval, choose the script from `run_train_retr_musique.sh`, `run_train_beam_retr.sh`, 
`run_train_2wiki.sh`, which aim at MuSiQue-Ans, HotpotQA and 2WikiMultihopQA respectively. Note that you should edit your actual url of data and model in the script. 

For open domain retrieval setting, we use the data produced by [MDR](https://github.com/facebookresearch/multihop_dense_retrieval/tree/main), and we format them in directory `fullwiki`, then train our Beam Retrieval using script `run_train_fullwiki_reranker`. 
## Downstream Reader
The code for the supervised downstream reader is in directory `qa`, while the code for LLMs is `llm_exp_long.py`.

## Results
All the results of retrieval and downstream reader are in directory `results`.

You can also obtain the scores through running `test_model_tmp.py` after training.
## Citation
```bibtex
@inproceedings{
zhang2024endtoend,
title={End-to-End Beam Retrieval for Multi-Hop Question Answering},
author={Jiahao Zhang and Haiyang Zhang and Dongmei Zhang and Yong Liu and Shen Huang},
booktitle={2024 Annual Conference of the North American Chapter of the Association for Computational Linguistics},
year={2024},
url={https://arxiv.org/abs/2308.08973}
}
```
