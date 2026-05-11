"""Synthetic LOFAR data generator (PCD v3 §7.4 platform substrate)."""

from fathom.synthetic.biologicals import (
    BiologicalInjectionPriors,
    SampledBiologicalInjection,
    inject_biologicals,
    load_biological_library,
)
from fathom.synthetic.generator import (
    C1_1_GENERATOR_VERSION,
    C1_2_GENERATOR_VERSION,
    GENERATOR_VERSION,
    generate_b1_clip,
    generate_c1_1_clip,
)
from fathom.synthetic.priors import (
    SampledTonalParameters,
    TonalParameterPriors,
    sample_n_sources,
    sample_tonal_parameters,
)
from fathom.synthetic.tonals import (
    inject_deterministic_tonal,
    inject_parameterized_tonal,
)
from fathom.synthetic.truth import compute_per_frame_truth, stft_frame_times_s

__all__ = [
    "BiologicalInjectionPriors",
    "C1_1_GENERATOR_VERSION",
    "C1_2_GENERATOR_VERSION",
    "GENERATOR_VERSION",
    "SampledBiologicalInjection",
    "SampledTonalParameters",
    "TonalParameterPriors",
    "compute_per_frame_truth",
    "generate_b1_clip",
    "generate_c1_1_clip",
    "inject_biologicals",
    "inject_deterministic_tonal",
    "inject_parameterized_tonal",
    "load_biological_library",
    "sample_n_sources",
    "sample_tonal_parameters",
    "stft_frame_times_s",
]