torchnmf.trainer
======================
.. automodule:: torchnmf.trainer

:mod:`torchnmf.trainer` is a package implementing various parameter updating algorithms for NMF, and is based on
the same optimizer interface from :mod:`torch.optim`.

Taking an update step
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because current available trainer reevaluate the function multiple times, a closure function is required in each step.
The closure should clear the gradients, compute output (or even the loss), and return it.

For :mod:`torchnmf.trainer.BetaMu`::

    for i in range(iterations):
        def closure():
            trainer.zero_grad()
            return target, model()
        trainer.step(closure)

For :mod:`torchnmf.trainer.SparsityProj`::

    for i in range(iterations):
        def closure():
            trainer.zero_grad()
            output = model()
            loss = loss_fn(output, target)
            return loss
        trainer.step(closure)


Algorithms
----------

.. autoclass:: BetaMu
    :members:
.. autoclass:: SparsityProj
    :members:
