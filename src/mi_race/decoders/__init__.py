"""decoders: learn to tell the symbols apart from the receiver signal

Every train function takes (X, y, settings, rng) and returns a trained model with 
predict_proba(X): one row per run, one column per symbol 0 ... N-1"""