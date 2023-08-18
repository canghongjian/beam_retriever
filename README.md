# Beam Retrieval: General End-to-End Retrieval for Multi-Hop Question Answering
This is the repository for our paper "Beam Retrieval: General End-to-End Retrieval for Multi-Hop Question Answering".

Our repository is under construction, feel free to contact us if you have any questions.

Our results have been published on [MuSiQue-Ans](https://leaderboard.allenai.org/musique_ans/submissions/public)

## Download Data and Model
We use three original datasets [MuSiQue-Ans](https://github.com/StonyBrookNLP/musique/), [HotpotQA](https://hotpotqa.github.io/) and [2WikiMultihopQA](https://github.com/Alab-NII/2wikimultihop) for our main 
experiments and three paritial datasets sampled by [IRCoT](https://github.com/StonyBrookNLP/ircot).

We use [DeBERTa](https://huggingface.co/microsoft/deberta-v3-base) as our backbone model.

## Beam Retrieval
The code for our Beam Retrieval is in directory `retrieval`. To train our Beam Retrieval, choose the script from `run_train_retr_musique.sh`, `run_train_beam_retr.sh`, 
`run_train_2wiki.sh`, which aim at MuSiQue-Ans, HotpotQA and 2WikiMultihopQA respectively. Note that you should edit your actual url of data and model in the script. 
## Downstream Reader
The code for the supervised downstream reader is in directory `qa`, while the code for gpt-3.5 is `gpt_turbo_exp.py`.

## Results
After training, you can obtain the scores through running `test_model_tmp.py`