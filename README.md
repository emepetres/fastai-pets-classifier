# fastai-pets-classifier

Pets classifier using fastai

```bash
conda create -f environment.yml
conda activate fastai
python -W ignore train_stage_1.py
python -W ignore train_stage_2.py
```

Questions raised from Lesson1 of walkwithfastai.com/revisited:

* How to save transforms so doesn't have to be done each time?
* It is possible to load model without having to recreate the learner?
* What lr_max=slice does?
* stage_2 does not really improves much the error rate

Small annoyances from Lesson1 of walkwithfastai.com/revisited:

* Seems set_seed is not working properly, as results are not reproducible
* How to show nbdev docs? <https://walkwithfastai.com/revisited/pets.html#untar_data>
* path.ls()[:3] the last part is not needed
* Learning rate really spikes around 10-3
* There is no deployment explained in Lesson 1, but diagrams shows it
