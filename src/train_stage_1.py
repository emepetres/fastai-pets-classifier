from learner import get_learner

if __name__ == "__main__":
    learn = get_learner()
    learn.fit_one_cycle(4)

    learn.save("stage_1")
