# SMOT: Simple Matter of Training

A repository of datascience / ai apis explored in far too much detail.

[Quick Links to API Tests](API.md)

You're probably most interested in the public api unittests in [api_tests](smot/api_tests):
  * [torch_api](smot/api_tests/torch_api)
    * torch
      * Tensors
        * [x] creation ops
        * [x] indexing, slicing, joining, mutating ops
      * [x] Generators
      * [ ] Random Sampling
      * [ ] Serialization
      * [ ] Parallelism
      * [ ] Locally disabling gradient computation
      * [ ] Math
      * [ ] Utilities
    * [ ] torch.???

These exist, but they're mostly empty at the moment.
  * [numpy_api](smot/api_tests/numpy_api)
  * [pandas_api](smot/api_tests/pandas_api)
 
## Test Naming

Generally, all test files have globally unique names. This makes IDE search functions
much smoother, at the cost of some local smurfing. For instance, despite being nested
in local contexts where their meaning is unambiguous, the following files exist:
  * [np_ones_test.py](smot/api_tests/numpy_api/creation/np_ones_test.py)
  * [torch_ones_test.py](smot/api_tests/torch_api/creation/torch_ones_test.py)

## Requirements

Packages
  * Apt Packages [apt.txt](apt.txt)
  * Snap Packages [snap.txt](snap.txt)
  * Python `virtualenv` Packages (`pip-compile` / `pip-sync`)
    - [requirements.in](requirements.in)
    - [requirements.txt](requirements.txt)


> Note: You need a working pytorch install. I'm using a cuda1.11 preview build,
> see the --link in requirements.txt, you may need to change this. I am open
> to making the bootstrapping code conditional on local torch needs. Hmu.
    
## Setup

  * Run `grind repo install_packages`
  * Run `grind repo bootstrap`
  * Verify the install:
    * Run `grind test --slow`

> If you encounter errors (I need a special version of torch for my cuda build)
> you may be able to satisfy them by adding constraints to the `.gitignore`d
> file `constraints.in`, here is mine:
> 
>     # pytorch with cu111 builds.
>     --find-links https://download.pytorch.org/whl/torch_stable.html
>     torch==1.10.1+cu111
>     torchvision==0.11.2+cu111
>     torchaudio==0.10.1+cu111


## Run All Tests

Pretty much everything gets run via:

`grind presubmit`

This will:
  * check all code for style violations
  * check all code for type violations
  * remove unused imports
  * reformat all code
  * run all tests that haven't been run


## Grind

See: [grind README](commands/README.md)

`grind` provides a hook to run many repository commands.

  * `grind check`
  * `grind format`
  * `grind test`
  * `grind fasttest`
  * `grind presubmit`

### Grind Tab Completion

Grind supports tab completion. If you'd like to set it up, add the following hook
to your bash environment:

```bash
source <$REPO_DIR>/tools/grind_completion
```


## Fancy Tests (eggs)

Tests are written using `hamcrest`, the most flexible pytest fluent testing framework I'm currently aware of
is [PyHamcrest](https://github.com/hamcrest/PyHamcrest)

Hamcrest is great, but it doesn't really dig into structural comparison
over tensor-like objects, so tests are also written in local hamcrest
extension libraries,  generally under the name `eggs`, `torch_eggs`, `np_eggs`, etc.

"Eggs and Ham", you see.

With some additional evolution, `eggs` may become a standalone package,
but I'll co-develop it against these api tests, and extract it once
I've got a solid usage corpus to iterate designs against.
