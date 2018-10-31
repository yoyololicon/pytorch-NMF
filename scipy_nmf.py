import numpy as np


def generalized_KL_divergence(V, V_tilde):
    return np.sum(V * np.log(V / V_tilde) - V + V_tilde)


def NMF(S, R, init_W=None, init_H=None, fix_W=False, fix_H=False, n_iter=50):
    """

    :param S: magnitude spectrogram
    :param R: number of templates
    :param init_W: initial weight
    :param init_H: initial activations
    :param fix_W:
    :param fix_H:
    :param n_iter:
    :return : W, H
    """

    K, M = S.shape

    if init_W is None:
        W = np.random.rand(K, R)
    else:
        W = init_W.copy()
    if init_H is None:
        H = np.random.rand(R, M)
    else:
        H = init_H.copy()

    for i in range(n_iter):
        V = W @ H
        loss = generalized_KL_divergence(S, V)
        print("Iter", "%d" % (i + 1), ", KL:" "%.4f" % loss)

        SonV = S / V
        # update W
        if not fix_W:
            W *= SonV @ H.T / H.T.sum(0)

        # update H
        if not fix_H:
            H *= W.T @ SonV / W.T.sum(1, keepdims=True)

    return W, H


def NMFD(S, R, T, init_W=None, init_H=None, fix_W=False, fix_H=False, n_iter=50):
    """

    :param S: magnitude spectrogram
    :param R: number of templates
    :param T: template length (number of frames)
    :param init_W: initial weight
    :param init_H: initial activations
    :param fix_W:
    :param fix_H:
    :param n_iter:
    :return: W, H
    """

    K, M = S.shape

    if init_W is None:
        W = np.random.rand(K, R, T)
    else:
        W = init_W.copy()
    if init_H is None:
        H = np.random.rand(R, M)
    else:
        H = init_H.copy()

    for i in range(n_iter):
        V_raw = np.tensordot(W, H, axes=(1, 0))
        V = V_raw[:, 0, :]
        for j in range(1, T):
            V[:, j:] += V_raw[:, j, :-j]

        loss = generalized_KL_divergence(S, V)
        print("Iter", "%d" % (i + 1), ", KL:" "%.4f" % loss)

        SonV = S / V
        # update W
        if not fix_W:
            expand_H = np.stack((np.roll(H, j, 1) for j in range(T)), axis=2)
            upper = np.tensordot(SonV, expand_H, axes=(1, 1))
            lower = np.swapaxes(expand_H, 0, 1).sum(0)
            W *= upper / lower

        # update H
        if not fix_H:
            expand_SonV = np.stack((np.roll(SonV, -j, 1) for j in range(T)), axis=2)
            upper = np.diagonal(np.tensordot(W, expand_SonV, axes=(0, 0)), axis1=1, axis2=3)
            lower = np.swapaxes(W, 0, 1).sum(1, keepdims=True)
            H *= np.mean(upper / lower, 2)

    return W, H


if __name__ == '__main__':
    import librosa
    import matplotlib.pyplot as plt
    from librosa import display

    y, sr = librosa.load(librosa.util.example_audio_file())
    S = np.abs(librosa.stft(y))

    R = 8
    T = 5
    comps, acts = NMFD(S, R=R, T=T, n_iter=50)
    # comps, acts = librosa.decompose.decompose(S, n_components=8, sort=True)

    display.specshow(np.log1p(comps[:, 0, :]), y_axis='log')
    plt.show()
    plt.imshow(np.log1p(acts), aspect='auto', origin='lower')
    plt.show()

    V_raw = np.tensordot(comps, acts, axes=(1, 0))
    V = V_raw[:, 0, :]
    for j in range(1, T):
        V[:, j:] += V_raw[:, j, :-j]
    display.specshow(np.log1p(V), y_axis='log')
    plt.show()
