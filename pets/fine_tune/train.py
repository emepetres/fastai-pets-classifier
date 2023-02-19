from pets.fine_tune.learner import get_vit_learner

if __name__ == "__main__":
    learn = get_vit_learner()
    learn.fine_tune(4, freeze_epochs=4)

    learn.export("models/vit_exported")
    learn.save("vit_saved")
