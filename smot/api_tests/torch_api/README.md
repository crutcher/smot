# pytorch api

See: [pytorch api docs](https://pytorch.org/docs/stable/torch.html)

# Major Gotchas

## `<f>(out=target)` can reallocate the data buffer

Many `torch` methods take an `out=<tensor>` mechanism, as a target for the data they generate. This mechanism is only
guaranteed to be data-in-place when the target tensor has the storage size and type needed to hold the data.

When the storage is incompatible, a new data tensor will be allocated, and attached to the target tensor.

Some api methods generate warnings, many do not.
