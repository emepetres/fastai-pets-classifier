from learner import get_learner


if __name__ == "__main__":
    learn = get_learner()
    learn.load("stage_1")

    learn.fit_one_cycle(4, lr_max=slice(1e-6, 1e-4))

    learn.save("stage_2")
