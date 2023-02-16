from learner import get_resnet_learner


if __name__ == "__main__":
    learn = get_resnet_learner()
    learn.load("stage_1")

    learn.unfreeze()
    learn.fit_one_cycle(4, lr_max=slice(1e-6, 1e-4))

    learn.save("stage_2")
