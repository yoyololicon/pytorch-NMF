Introduction by Example
=======================

.. contents::
    :local:

Background
-----------

The goal of Non-negative Matrix Factorization (NMF) is, given a N by M non-negative matrix :obj:`V`, find a R by M
non-negative matrix :obj:`H` (typically called activation matrix) and a N by R non-negative matrix :obj:`W` (
typically called template matrix) such that their matrix product :obj:`WH` approximate :obj:`V` to some degree.
Generally, R is chosen to be smaller than ``min(N, M)``, which implies that high-dimensional data :obj:`V` can be reduced
to some low-dimensional space.

Basic Non-negative Matrix Factorization
------------------------------------------

Let's see how PyTorch NMF work in action!

First, assuming that we have a target matrix :obj:`V` with shape :obj:`[64, 1024]`:

.. code-block:: python

    V.size()
    >>> torch.Size([64, 1024])

Second, we need to construct a NMF instance by giving the shape of the target matrix and a latent order :obj:`R`.
We use :obj:`R = 10`:

.. code-block:: python

    import torch
    from torchnmf.nmf import NMF

    model = NMF(V.t().shape, rank=10)
    mode.W.size()
    >>> torch.Size([64, 10])
    mode.H.size()
    >>> torch.Size([1024, 10])

Now our model has two attributes, :obj:`W` and :obj:`H` with the shape defined in the previous section.

.. Note::
    The :obj:`H` is actually presented as transposed matrix in our implementation.

Then, fitting two matrix to our target data::

    mode.fit(V.t())

The reconstructed matrix is the matrix product of the two trained matrix::

    WH = model.W @ model.H.t()

Or you can just simply call the model and it will done by itself::

    WH = model()


Training on GPU
---------------

If you have NVIDIA GPU installed machine and have installed CUDA, you can try moving your model and target matrix to GPU,
and see how it speed up the fitting process:

.. code-block:: python

    V = V.cuda().t()
    model = model.cuda()
    model.fit(V)

In PyTorch NMF, we implemented different kinds of NMF by inheriting and extending :obj:`torch.nn.Module` object, so you
can treat them just like any other PyTorch Module (ex: moving among different devices, casting to different data type... etc.)


Model Concatenation
---------------------

Started at version 0.3, you can now combine different NMF module into a single module, and train it in an end-to-end fashion.

Let's use the previous example again. Instead of factorize matrix :obj:`V` into 2 matrix, we factorize it into 4 matrix.
That is:

.. math::

   V \approx W_1W_2W_3H

It's actually just chaining 3 NMF module all together, with 2 of them use the output from other NMF as their activation matrix.

Here is how you do it:

.. code-block:: python

    import torch
    import torch.nn as nn
    from torchnmf.nmf import NMF

    class Chain(nn.Module):
        def __init__(self):
            super().__init__()
            self.nmf1 = NMF((1024, 10), rank=4)
            self.nmf2 = NMF(W=(24, 10))
            self.nmf3 = NMF(W=(64, 24))

        def forward(self):
            WH = self.nmf1()
            WWH = self.nmf2(H=WH)
            WWWH = self.nmf3(H=WWH)
            return WWWH

    model = Chain()
    output = model()

You can also use :obj:`torch.nn.Sequential` to construct this kind of chaining model:

.. code-block:: python

    model = nn.Sequential(
        NMF((1024, 10), rank=4),
        NMF(W=(24, 10)),
        NMF(W=(64, 24))
    )
    # In newer version of PyTorch at least one input should be given
    # We can just give it `None`
    output = model(None)

To fit the model, instead of calling class method ``fit``, you now need to construct a NMF trainer:

.. code-block:: python

    from torchnmf.trainer import BetaMu

    trainer = BetaMu(model.parameters())

To update parameters, you need to call ``step()`` function in every iteration:

.. code-block:: python

    epochs = 200

    for e in range(epochs):
        def closure():
            trainer.zero_grad()
            return V.t(), model(None)
        trainer.step(closure)