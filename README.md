# fastai-pets-classifier

Pets classifier samples with/without fastai. Also a second sample using raw pytorch for data, model and optimizer, except the learner.

## Manual fine-tune sample

Simple sample that shows how fine-tunning is done by fastai in two stages.

* stage_1 & stage_2 is equivalent of learn.fine_tune(4, freeze_epochs=4). lr and other hyps can be tweaked as well.
* lr_max=slice(..) actually allows you to use different lr for body and head

```bash
conda create -f environment.yml
conda activate fastai
python -W ignore -m pets.fine_tune.train_stage_1
python -W ignore -m pets.fine_tune.train_stage_2
```

## Pytorch

Training using raw pytorch data api, our own model, and a pytorch optimizer. From fastai we are just using the learner.

**FIXME**: It seems to not be using the GPU.
The reason is that `Normalize` transform is being done in the items transforms side. TODO do Normalize as a batch transform.

**TODO**: Check if model is being correctly freezed.

```bash
conda create -f environment.yml
conda activate fastai
python -W ignore -m pets.pytorch.train
```

## Unknown images support

Support to return no label if the image is not recognized, by turning single label classification into multilabel with an accuracy threshold of 0.95.

```bash
conda create -f environment.yml
conda activate fastai
python -W ignore -m pets.unknown_imgs.train
```

## Deployment

Shows how to deploy with and without using fastai. The former is easier, but fastai is too unstable to be used as inference framework, as with time it is quite hard to replicate exact environment, not mentioning upgrading fastai code to a new version (many breaking changes).

```bash
conda create -f environment.yml
conda activate fastai
python -W ignore -m pets.deploy.train
python -W ignore -m pets.deploy.with_fastai
```

**TODO**: deployment without fastai

**TODO**: deployment on hugging face
