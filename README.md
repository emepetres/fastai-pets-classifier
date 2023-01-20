# fastai-pets-classifier

Pets classifier using fastai

## NOTES:

* stage_1 & stage_2 is equivalent of learn.fine_tune(4, freeze_epochs=4)
* lr_max=slice(..) actually allows you to use different lr for body and head
* learner creation should be refactored to its own class, so you can create and load weights to it in stage 2


## Run it

```bash
conda create -f environment.yml
conda activate fastai
python -W ignore train_stage_1.py
python -W ignore train_stage_2.py
```
