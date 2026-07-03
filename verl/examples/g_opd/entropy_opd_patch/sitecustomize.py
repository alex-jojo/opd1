"""No-op placeholder.

This directory used to rely on sitecustomize for the Entropy OPD experiment,
but importing training patches in every Ray Python process is too invasive.
The current implementation is guarded directly in verl code by
ENTROPY_OPD_ENABLE.
"""
