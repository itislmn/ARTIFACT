"""
methods - every DoA method of the experiment, one module each; CORE is the set shown in the main figures, the rest is
in the appendix; FDR_CONTROLLED have a guarantee for this fixed steering matrix, WITH_ALPHA also Model-X knockoff+.

    from estimation.methods import doa_methods, labels, CORE
"""
import importlib

MODULES = ["bartlett", "capon", "music", "esprit", "cfar", "bh", "by", "model_x_knockoffs", "fixed_x_knockoffs",
           "ctrex.ctrex_t_lars", "ctrex.ctrex_t_lasso", "ctrex.ctrex_elastic_net", "ctrex.ctrex_t_omp",
           "ctrex.ctrex_t_ols", "ctrex.ctrex_t_mp", "ctrex.ctrex_t_gp", "ctrex.ctrex_t_acgp",
           "ctrex.ctrex_dependency_aware", "trex_real", "infinirex", "dnn"]
CORE = ["bartlett", "music", "esprit", "os_cfar", "bh", "knockoffs", "knockoffs_lowdim", "ctrex_tlars",
        "ctrex_tomp", "trex_tlars", "infinirex", "infinirex_radar", "dnn"]
FDR_CONTROLLED = ["bh", "by", "knockoffs_lowdim", "ctrex_tlars", "ctrex_tlasso", "ctrex_ten", "ctrex_tomp",
                  "ctrex_tols", "ctrex_tmp", "ctrex_tgp", "ctrex_tacgp", "ctrex_tlars_da", "ctrex_tomp_da",
                  "trex_tlars", "trex_tomp", "infinirex",
                  "infinirex_radar"]
WITH_ALPHA = FDR_CONTROLLED + ["knockoffs", "knockoffs_offset0", "knockoffs_lowdim_offset0"]


def specs():
    return [s for m in MODULES for s in importlib.import_module(f"estimation.methods.{m}").METHODS]


def doa_methods(names=None):
    every = specs()
    if not names:
        return every
    wanted = set(names)
    unknown = wanted - {s.name for s in every} - {s.family for s in every}
    if unknown:
        raise SystemExit(f"unknown method(s) {sorted(unknown)}; known: {[s.name for s in every]}")
    return [s for s in every if s.name in wanted or s.family in wanted]


def labels():
    return {s.name: s.label for s in specs()}
