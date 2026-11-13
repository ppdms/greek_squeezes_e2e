"""The page network: one model from a squeeze's detections to which of them are letters and which letters they are.

Every detection of the page is a token. A DINOv3 ViT-S/16 encodes the token's crop of the relit relief; its detector
outputs and its shape and position in the deskewed letter-height frame join that encoding; a transformer then lets
every token attend to its nearest neighbours on the page, with the attention biased by their relative geometry. One
head gives each token 27 letters or none. Trained end to end against the annotations matched one-to-one to the
detections, the network decides in one latent space which letter a detection shows, whether it is a letter at all
given its neighbours, and part of what a language model knows (which letter fits the row). The row decoder reads its
output: the probability of none sets each token's keep probability, the 27 letters its emission.

    pages.py     tokens of a page: geometry, neighbours, targets
    model.py     the network
    train.py     training one chain's network on the other folds' detections
    infer.py     reading a chain's pages
    decode.py    the row decoder over the network's output
    stage.py     command line
"""
