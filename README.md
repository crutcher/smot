# SMOT: Simple Matter of Training

AI api experiments.

## Requirements

Packages
  * Apt Packages [apt.txt](./apt.txt)
  * Snap Packages [snap.txt](./snap.txt)
  * Python `virtualenv` Packages (`pip-compile` / `pip-sync`)
    - [requirements.in](./requirements.in)
    - [requirements.txt](./requirements.txt)


## Setup

  * Run `./grind repo install_packages`
  * Run `./grind repo bootstrap`


## Grind

`./grind` provides a hook to run many different repository commands.

  * `./grind check`
  * `./grind format`
  * `./grind test`
  * `./grind fasttest`
  * `./grind presubmit`

See: [grind README](commands/README.md)

