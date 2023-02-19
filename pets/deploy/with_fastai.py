from fastai.vision.all import load_learner, Learner, untar_data, URLs, get_image_files


def predict1(learn: Learner, fname: str):
    """Using predict from fastai to predict one item."""
    return learn.predict(fname)


def predict2(learn: Learner, fname: str):
    """Using get_preds to predict. It can be used for predicting more than one item."""
    learn.dls.cuda()  # NOTE: by default learner is loaded to cpu
    dl = learn.dls.test_dl(
        [fname],
        num_workers=0,  # NOTE: it's important to disable multiprocessing
    )
    preds = learn.get_preds(dl=dl)[0]
    softmax = preds.softmax(dim=1)
    argmax = preds.argmax(dim=1)
    labels = [learn.dls.vocab[pred] for pred in argmax]
    return (labels, argmax, softmax)


if __name__ == "__main__":
    path = untar_data(URLs.PETS) / "images"
    fnames = get_image_files(path)

    learn = load_learner("models/vit_exported")  # NOTE: cpu=False loads on gpu

    print(predict1(learn, fnames[0])[0])
    print(predict2(learn, fnames[0])[0])
