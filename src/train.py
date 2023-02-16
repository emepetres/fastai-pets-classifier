from learner import get_vit_learner

if __name__ == "__main__":
    learn = get_vit_learner()
    learn.fine_tune(1)

    learn.export("models/exported_fastai")
    learn.save("exported_model")
