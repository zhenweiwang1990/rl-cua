"""VLM (Vision-Language Model) inference module.

This module re-exports the shared VLMInference from gbox-cua package.
Install with: pip install git+https://github.com/babelcloud/gbox-cua.git
"""

from gbox_cua.vlm_inference import VLMInference

__all__ = ["VLMInference"]
