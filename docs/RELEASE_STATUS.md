# Initial source release — v0.1.1

Released on 2026-10-16 from the maintainer-selected KBQA-R1/GAPD working tree. The source snapshot includes local modifications; its base Git commit is `e8b6e1ef662a978c01cffb8bb682c0d9e0b79f19`.

Repository: https://github.com/sunxin000/GAPD

## Included

- GAPD implementation, KBQA execution modules and the customized VERL fork.
- Editable package metadata, Hydra resources and Python 3.12+ requirement.
- Portable training/evaluation entry points, backend/model configuration and dry-run mode.
- Gold-trace preparation/validation, a portable SFT launcher and citation metadata.
- Root Apache-2.0 license, third-party headers and upstream license/notice texts.
- CPU unit tests and a static release audit command.

## Validation

Thirteen CPU tests pass, Python and shell syntax checks pass, and wheel/sdist package builds include the required code, Hydra configurations and license files.

## External resources

Model weights, benchmark datasets and Freebase are configured separately as described in `SETUP.md` and `DATA.md`. The root license follows Apache-2.0, as declared in the original source README; third-party components retain their own terms.
