from raw_learner import get_learner

if __name__ == "__main__":
    learn = get_learner()
    model = learn.model

    # freeze the body
    for layer in list(model.children())[:-1]:
        if hasattr(layer, "requires_grad_"):
            layer.requires_grad_(False)

    learn.fit_one_cycle(5, 1e-3)
    learn.save("raw")

    # now we could perform a second stage for complete model
