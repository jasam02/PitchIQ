"""Soccer-specific tracking layer for the GPU prototype.

Everything in this package is torch-free (numpy + OpenCV only) so it can be
unit-tested without CUDA. The detector and BoT-SORT live in pipeline.py.

Flow per frame:
    detections -> pitch.PitchDetector -> relevance.assess (foot point, zones,
    audience, size) -> local tracker (BoT-SORT) -> appearance descriptors ->
    teams (kit model, role votes) -> identity.IdentityManager (global players)

Local track IDs are temporary. Global identities (A-07, GK-1, REF-1) persist
for the whole run and can be continued into the next run.
"""
