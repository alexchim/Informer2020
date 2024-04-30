Project
COMP4434 BIG DATA ANALYTICS

Quick start
SVR: python -u main_informer.py --model SVR --data ETTh1 --attn prob --freq h
RFR: python -u main_informer.py --model RFR --data ETTh1 --attn prob --freq h
BasicNN: python -u main_informer.py --model basic --data ETTh1 --attn prob --freq h
CustomNN: python -u main_informer.py --model custom --data ETTh1 --attn prob --freq h

the dataset should be contain in ./data/ETT/ETTh1.csv

the result after running will be in ./result and the model will be in ./checkpoint
