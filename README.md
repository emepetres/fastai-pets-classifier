# fastai-pets-classifier

Pets classifier samples with/without fastai. Also a second sample using raw pytorch for data, model and optimizer, except the learner.

## Fine-tune sample

Simple sample that shows how fine-tunning is done by fastai in two stages.

* stage_1 & stage_2 is equivalent of learn.fine_tune(4, freeze_epochs=4). lr and other hyps can be tweaked as well.
* lr_max=slice(..) actually allows you to use different lr for body and head

```bash
conda create -f environment.yml
conda activate fastai
python -W ignore -m pets.fine_tune.train_stage_1
python -W ignore -m pets.fine_tune.train_stage_2
```

## Raw

Training using raw pytorch data api, our own model, and a pytorch optimizer. From fastai we are just using the learner.

**FIXME**: It seems to not be using the GPU.

**FIXME**: Are we really freezing the model?

```bash
conda create -f environment.yml
conda activate fastai
python -W ignore -m pets.fine_tune.train_stage_1
python -W ignore -m pets.fine_tune.train_stage_2
```
