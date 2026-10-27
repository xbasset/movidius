"""Rank-one PCA and the author's flattened mask morphology on the VPU."""

import numpy as np


def pca_mask(g, raw, iterations=64):
    n, width = raw.shape[1:]
    rows = g.const("pca/rows", [1], "I32")
    cols = g.const("pca/cols", [2], "I32")
    mean = g.node("pca/mean", "ReduceMean", [raw, rows], (1, 1, width), {"keep_dims": "true"})
    # Materialize the dynamic vector broadcast; the old kernel corrupts its tail.
    expanded = g.gemm("pca/expanded_mean", g.const("pca/ones", np.ones((1, n, 1))), mean)
    centered = g.eltwise("pca/center", raw, expanded, "sub")
    scaled = g.power("pca/scaled", centered, scale=1 / 128)
    transposed = g.permute("pca/transpose", scaled, (0, 2, 1))
    initial = np.random.default_rng(0).standard_normal((1, width, 1))
    initial /= np.linalg.norm(initial)
    vector = g.const("pca/initial", initial)
    for i in range(iterations):
        projection = g.gemm(f"pca/{i}/project", scaled, vector)
        vector = g.gemm(f"pca/{i}/update", transposed, projection)
        square = g.power(f"pca/{i}/square", vector, power=2)
        total = g.node(f"pca/{i}/sumsq", "ReduceSum", [square, rows], (1, 1, 1), {"keep_dims": "true"})
        invnorm = g.power(f"pca/{i}/invnorm", total, power=-.5)
        vector = g.eltwise(f"pca/{i}/normalize", vector, invnorm, "prod")
    absolute = g.eltwise("pca/abs", vector, g.power("pca/negative_vector", vector, scale=-1), "max")
    maximum = g.node("pca/max_abs", "ReduceMax", [absolute, rows], (1, 1, 1), {"keep_dims": "true"})
    largest = g.eltwise("pca/largest", absolute, maximum, "equal")
    signed = g.eltwise("pca/largest_signed", vector, largest, "prod")
    coefficient = g.node("pca/coefficient", "ReduceSum", [signed, rows], (1, 1, 1), {"keep_dims": "true"})
    positive_sign = g.eltwise("pca/sign_test", coefficient, g.const("pca/zero", np.zeros((1, 1, 1))), "greater")
    sign = g.power("pca/sign", positive_sign, scale=2, shift=-1)
    vector = g.eltwise("pca/oriented", vector, sign, "prod")
    pc = g.gemm("pca/pc", centered, vector)
    positive = g.eltwise("pca/positive", pc, g.const("pca/threshold", np.full((1, 1, 1), 10)), "greater")
    negative = g.eltwise("pca/negative", pc, g.const("pca/negative_threshold", np.full((1, 1, 1), -10)), "less")
    grid = g.reshape("pca/grid", positive, (1, 32, 32))
    center = g.crop("pca/center_rows", grid, 1, 6, 19)
    center = g.crop("pca/center_cols", center, 2, 6, 19)
    axes = g.const("pca/grid_axes", [1, 2], "I32")
    count = g.node("pca/center_count", "ReduceSum", [center, axes], (1, 1, 1), {"keep_dims": "true"})
    flip = g.eltwise("pca/flip", count, g.const("pca/center_threshold", np.full((1, 1, 1), .35 * 19 * 19)), "less_equal")
    keep = g.power("pca/keep", flip, scale=-1, shift=1)
    mask = g.eltwise("pca/combine", g.eltwise("pca/kept", positive, keep, "prod"),
                     g.eltwise("pca/flipped", negative, flip, "prod"), "sum")

    def morphology(name, x, operation, border):
        pad = g.const(name + "/border", np.full((1, 1, 1), border))
        left = g.node(name + "/left", "Concat", [pad, g.crop(name + "/left_crop", x, 1, 0, n-1)], x.shape, {"axis": 1})
        right = g.node(name + "/right", "Concat", [g.crop(name + "/right_crop", x, 1, 1, n-1), pad], x.shape, {"axis": 1})
        return g.eltwise(name, g.eltwise(name + "/first", left, x, operation), right, operation)

    mask = morphology("pca/dilate", mask, "max", 0)
    mask = morphology("pca/close_dilate", mask, "max", 0)
    return morphology("pca/close_erode", mask, "min", 1)
