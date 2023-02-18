from pets.unknown_imgs.learner import get_learner

if __name__ == "__main__":
    learn = get_learner()
    learn.fit_one_cycle(4, 2e-3)

    learn.save("unknown_items")
