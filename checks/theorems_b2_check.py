"""Assert the B.20--B.34 dictionary formulas and review counterexamples.

Run from the repository root with:
    uv run --with numpy python -I checks/theorems_b2_check.py

Small deterministic CPU checks; no data downloads, training, or extra packages.
"""

import itertools
import math

import numpy as np


def balance(kernel):
    """Symmetric Sinkhorn scaling, with an asserted numerical residual."""
    scale = 1 / np.sqrt(kernel.sum(axis=1))
    for _ in range(20000):
        row_sums = scale * (kernel @ scale)
        if np.max(np.abs(row_sums - 1)) < 3e-14:
            break
        scale /= np.sqrt(row_sums)
    operator = scale[:, None] * kernel * scale[None, :]
    assert np.max(np.abs(operator.sum(axis=1) - 1)) < 1e-12
    return operator


def indicators(labels):
    labels = np.asarray(labels)
    groups = np.unique(labels)
    return np.column_stack(
        [(labels == group) / np.sqrt(np.sum(labels == group)) for group in groups]
    )


def partitions(n_sites, n_groups):
    """Every unordered partition once, as a restricted growth string."""
    def extend(prefix, used):
        if len(prefix) == n_sites:
            if used == n_groups:
                yield np.array(prefix)
            return
        for group in range(min(used + 1, n_groups)):
            yield from extend(prefix + [group], max(used, group + 1))

    yield from extend([0], 1)


def centroid_loss(data, labels):
    return sum(
        np.sum((data[labels == group] - data[labels == group].mean(axis=0)) ** 2)
        for group in np.unique(labels)
    )


def line_loss(data, labels):
    total = 0.0
    for group in np.unique(labels):
        moment = data[labels == group].T @ data[labels == group]
        total += np.trace(moment) - np.linalg.eigvalsh(moment)[-1]
    return total


def signed_frame_error(frame, target):
    errors = []
    for permutation in itertools.permutations(range(target.shape[1])):
        candidate = target[:, permutation]
        signs = np.sign(np.sum(frame * candidate, axis=0))
        signs[signs == 0] = 1
        errors.append(np.linalg.norm(frame - candidate * signs))
    return min(errors)


def rotate_two(frame, angle):
    cosine, sine = np.cos(angle), np.sin(angle)
    return frame @ np.array([[cosine, -sine], [sine, cosine]])


def global_max_ipr_two(frame):
    """For two columns, objective = constant + B cos(4t) + C sin(4t).

    Reflections only change a column sign, so optimizing rotations covers O(2).
    """
    objective = lambda angle: np.sum(rotate_two(frame, angle) ** 4)
    constant = (objective(0) + objective(np.pi / 4)) / 2
    cosine_term = (objective(0) - objective(np.pi / 4)) / 2
    sine_term = objective(np.pi / 8) - constant
    optimum_angle = np.arctan2(sine_term, cosine_term) / 4
    result = rotate_two(frame, optimum_angle)
    assert abs(np.sum(result**4) - constant - np.hypot(cosine_term, sine_term)) < 1e-12
    for angle in np.linspace(0, np.pi / 2, 101):
        assert objective(angle) <= np.sum(result**4) + 1e-12
    return result


def check_cluster_bounds():
    rng = np.random.default_rng(20261107)
    sizes = np.array([20, 30, 40])
    n_groups, n_sites = len(sizes), sum(sizes)
    labels = np.repeat(np.arange(n_groups), sizes)
    target = indicators(labels)
    size_ratio = max(
        sum(np.sqrt(sizes[other] / sizes[group]) for other in range(n_groups) if other != group)
        for group in range(n_groups)
    )
    for noise in [0.0, 0.035]:
        data = np.eye(6)[labels] + noise * rng.normal(size=(n_sites, 6))
        data /= np.linalg.norm(data, axis=1)[:, None]
        gram = data @ data.T
        same = labels[:, None] == labels[None, :]
        epsilon = max(0.0, 1 - gram[same].min())
        separation = 1 - gram[~same].max()
        for beta in [6.0, 9.0, 12.0]:
            operator = balance(np.exp(beta * (gram - 1)))
            values, vectors = np.linalg.eigh(operator)
            values, vectors = values[::-1], vectors[:, ::-1]
            sine = np.linalg.norm((np.eye(n_sites) - target @ target.T) @ vectors[:, :n_groups], 2)
            gap = values[n_groups - 1] - values[n_groups]
            within = np.exp(-beta * epsilon)
            across = np.exp(-beta * separation)
            leakage_ratio = across * size_ratio / within
            leakage = leakage_ratio / np.sqrt(within * (1 - leakage_ratio))
            mixing = within**2 / (1 + across * size_ratio) ** 2
            error = 2 * leakage
            if noise == 0:
                gap_bound, angle_bound = 1 - 2 * across * size_ratio, 0.0
                assert values[n_groups - 1] >= gap_bound - 1e-12
                assert abs(values[n_groups]) < 1e-12 and sine < 1e-12
            else:
                assert 2 * error < mixing
                gap_bound, angle_bound = mixing - 2 * error, error / (mixing - error)
                assert values[n_groups - 1] >= 1 - error - 1e-12
                assert values[n_groups] <= 1 - mixing + error + 1e-12
                assert gap >= gap_bound - 1e-12 and sine <= angle_bound + 1e-12
                # Intermediate balancing inequalities, not just the final gap.
                assert np.max(np.sum(np.where(same, 0, operator), axis=1)) <= leakage + 1e-12
                for group, size in enumerate(sizes):
                    block = operator[np.ix_(labels == group, labels == group)]
                    assert np.min(block) >= mixing / size - 1e-12
            print(
                f"noise={noise:.3f} beta={beta:g} gap={gap:.8f} gap_lower={gap_bound:.8f} "
                f"sin_theta={sine:.8g} sine_upper={angle_bound:.8g}"
            )
    for _ in range(100):
        rotation, _ = np.linalg.qr(rng.normal(size=(n_groups, n_groups)))
        assert np.sum((target @ rotation) ** 4) <= np.sum(target**4) + 1e-14


def check_localization_and_certificate():
    rng = np.random.default_rng(1097)
    for sizes, sine in [([3, 4], 1e-4), ([3, 4], 1e-2), ([2, 30], 1e-3)]:
        target = indicators(np.repeat(np.arange(2), sizes))
        perturbation = rng.normal(size=target.shape)
        perturbation -= target @ (target.T @ perturbation)
        perturbation, _ = np.linalg.qr(perturbation)
        frame = np.sqrt(1 - sine**2) * target + sine * perturbation
        localized = global_max_ipr_two(frame)
        observed = signed_frame_error(localized, target)
        frame_bound = np.sqrt(4) * sine
        objective_bound = 4 * np.sqrt(2) * frame_bound
        deficit_bound = 2 * max(sizes) * objective_bound
        threshold = 1 / (16 * np.sqrt(2) * 2 * max(sizes))
        assert (deficit_bound < 0.5) == (sine < threshold)
        if deficit_bound < 0.5:
            assert observed <= frame_bound + np.sqrt(2 * deficit_bound) + 1e-12
        print(f"B.23 sizes={sizes} tau={sine:g} applicable={deficit_bound < .5} observed_error/tau={observed/sine:.6f}")
    # B.24: compute the certificate without using the planted labels in the search.
    target = indicators([0, 0, 0, 1, 1, 1, 1])
    perturbation = rng.normal(size=target.shape)
    perturbation -= target @ (target.T @ perturbation)
    perturbation, _ = np.linalg.qr(perturbation)
    sine = 0.05
    frame = np.sqrt(1 - sine**2) * target + sine * perturbation
    projector = frame @ frame.T
    candidates = [(np.linalg.norm(projector - (u := indicators(labels)) @ u.T, 2), u)
                  for labels in partitions(7, 2)]
    candidates.sort(key=lambda pair: pair[0])
    distance, recovered = candidates[0]
    assert distance < 1 / 14 and candidates[1][0] > distance + 1e-12
    assert np.linalg.norm(recovered @ recovered.T - target @ target.T) < 1e-12
    assert abs(distance - sine) < 1e-12
    values, vectors = np.linalg.eigh(recovered.T @ projector @ recovered)
    localized = projector @ recovered @ (vectors * values**-0.5) @ vectors.T
    assert np.linalg.norm(localized.T @ localized - np.eye(2)) < 1e-12
    assert np.linalg.norm(localized - recovered) <= np.sqrt(4) * distance + 1e-12


def check_loading_and_sparse_baselines():
    # Sol's exact-ray counterexample: normalized reconstruction versus raw loadings.
    raw = np.array([[1.0, 0], [3, 0], [0, 2], [0, 4]])
    normalized = raw / np.linalg.norm(raw, axis=1)[:, None]
    labels = np.array([0, 0, 1, 1])
    target = indicators(labels)
    normalized_dictionary = normalized.T @ target / np.sqrt(2)
    raw_dictionary = raw.T @ target / np.sqrt(2)
    assert np.allclose(normalized_dictionary, np.eye(2))
    assert np.allclose(raw_dictionary, np.diag([2, 3]))
    assert abs(np.linalg.norm(raw_dictionary - normalized_dictionary) - np.sqrt(5)) < 1e-12
    assert min(centroid_loss(normalized, candidate) for candidate in partitions(4, 2)) == 0
    operator = balance(np.exp(4 * (normalized @ normalized.T - 1)))
    _, vectors = np.linalg.eigh(operator)
    assert np.linalg.norm(vectors[:, -2:] @ vectors[:, -2:].T - target @ target.T) < 1e-12
    print(f"B.26 wrong-loading-view dictionary difference={np.linalg.norm(raw_dictionary-normalized_dictionary):.8f}")
    # Opus's cluster-size obstruction, proved by an explicitly better partition.
    size, epsilon, separation = 8, 0.0275, 0.1
    data = np.array([[1., 0, 0, 0]] * 4
                    + [[1-epsilon, np.sqrt(1-(1-epsilon)**2), 0, 0]] * 4
                    + [[0, 0, 1, 0], [0, 0, 1-separation, np.sqrt(1-(1-separation)**2)]])
    planted = np.array([0] * 8 + [1, 2])
    alternative = np.array([0] * 4 + [1] * 4 + [2, 2])
    assert abs(centroid_loss(data, planted) - size * epsilon / 2) < 1e-12
    assert abs(centroid_loss(data, alternative) - separation) < 1e-12
    assert centroid_loss(data, alternative) < centroid_loss(data, planted)
    # B.27: two angular clusters, including the exact orthogonal second ray.
    theta = 0.05
    data = np.array([[1., 0, 0], [1., 0, 0], [np.cos(theta), np.sin(theta), 0]]
                    + [[0, 0, 1.]] * 3)
    planted = np.array([0] * 3 + [1] * 3)
    planted_loss = line_loss(data, planted)
    wrong_losses = [line_loss(data, candidate)
                    for groups in [1, 2] for candidate in partitions(6, groups)
                    if not np.array_equal(candidate, planted)]
    assert planted_loss < 1 and min(wrong_losses) >= 1 - 1e-12
    moment = data[:3].T @ data[:3]
    _, vectors = np.linalg.eigh(moment)
    atom = vectors[:, -1]
    centroid = data[:3].mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    angle = np.arctan2(np.linalg.norm(np.cross(atom, centroid)), abs(atom @ centroid))
    assert 4.6e-6 < angle < 4.7e-6
    assert abs(angle / (theta**3 / 27) - 1) < 0.01
    print(f"B.27 two-cluster atom/centroid angle={angle:.10g} radians")
    # B.28: signed codes can merge antipodes; nonnegative codes cannot.
    amplitudes = np.array([1., 2., -1., -3.])
    atoms = np.array([[1., -1.]])
    codes = np.column_stack([np.maximum(amplitudes, 0), np.maximum(-amplitudes, 0)])
    assert np.all(np.count_nonzero(codes, axis=1) == 1)
    assert np.array_equal(codes @ atoms.T, amplitudes[:, None])
    assert np.min([np.sum(np.minimum(amplitudes, 0)**2), np.sum(np.maximum(amplitudes, 0)**2)]) == 5
    # B.29: a merely finite, large ratio bound does not create separated supports.
    same_support = np.array([[100., 1.], [1., 100.]])
    same_support /= np.linalg.norm(same_support, axis=1)[:, None]
    assert same_support[0] @ same_support[1] < same_support[0] @ np.array([1., 0.])
    # B.30: continuum of positive, orthogonal dictionaries, identical observations.
    raw = np.array([1., 2., 3.])[:, None] * np.array([[1., 1.]])
    reference_kernel = np.exp(2 * (raw / np.linalg.norm(raw, axis=1)[:, None]) @
                              (raw / np.linalg.norm(raw, axis=1)[:, None]).T)
    for angle in [0., .1, .2, .3]:
        cosine, sine = np.cos(angle), np.sin(angle)
        dictionary = np.array([[cosine, -sine], [sine, cosine]])
        codes = raw @ dictionary
        assert np.all(codes > 0)
        reconstructed = codes @ dictionary.T
        assert np.allclose(reconstructed, raw)
        normalized = reconstructed / np.linalg.norm(reconstructed, axis=1)[:, None]
        assert np.allclose(np.exp(2 * normalized @ normalized.T), reference_kernel)
        assert np.allclose(balance(np.exp(2 * normalized @ normalized.T)), balance(reference_kernel))
        assert np.allclose(reconstructed.T @ indicators([0, 0, 0]), raw.T @ indicators([0, 0, 0]))


def check_diagnostics_hierarchy_sampling():
    rng = np.random.default_rng(519)
    for n_groups, radius_squared, beta in [(2, .1, .3), (3, .45, 5), (4, .01, .01)]:
        data = []
        for group in range(n_groups):
            for sign in [1, -1]:
                row = np.zeros(2 * n_groups)
                row[group] = np.sqrt(1 - radius_squared)
                row[n_groups + group] = sign * np.sqrt(radius_squared)
                data.append(row)
        data = np.array(data)
        target = indicators(np.repeat(np.arange(n_groups), 2))
        kernel = np.exp(beta * data @ data.T)
        operator = kernel / kernel.sum(axis=1)[0]
        _, vectors = np.linalg.eigh(operator)
        assert np.linalg.norm(vectors[:, -n_groups:] @ vectors[:, -n_groups:].T - target @ target.T) < 1e-10
    sizes = np.array([2, 3, 7])
    target = indicators(np.repeat(np.arange(3), sizes))
    diagonal = np.sum(target**2, axis=1)
    assert abs(np.sum((diagonal / 3)**2) - np.sum(1/sizes) / 9) < 1e-14
    centered_diagonal = diagonal - 1/12
    assert abs(np.sum((centered_diagonal / 2)**2) - (np.sum(1/sizes) - 5/12) / 4) < 1e-14
    for rank in range(1, 13):
        frame, _ = np.linalg.qr(rng.normal(size=(12, rank)))
        ipr = np.sum((np.sum(frame**2, axis=1) / rank)**2)
        assert 1/12-1e-14 <= ipr <= 1/rank+1e-14
    n_coarse, n_fine, repeats = 2, 3, 4
    fine = np.repeat(np.arange(n_coarse * n_fine), repeats)
    coarse = fine // n_fine
    gram = np.where(fine[:, None] == fine[None, :], 1.,
                    np.where(coarse[:, None] == coarse[None, :], .99, .2))
    for beta in [5., 300.]:
        operator = balance(np.exp(beta * (gram - 1)))
        values = np.linalg.eigvalsh(operator)[::-1]
        fine_weight, coarse_weight = np.exp(-beta * .01), np.exp(-beta * .8)
        denominator = 1 + (n_fine-1)*fine_weight + n_fine*(n_coarse-1)*coarse_weight
        coarse_value = 1 - n_coarse*n_fine*coarse_weight/denominator
        fine_value = (1-fine_weight)/denominator
        expected = np.array([1.] + [coarse_value]*(n_coarse-1)
                            + [fine_value]*(n_coarse*(n_fine-1))
                            + [0.]*(len(fine)-n_coarse*n_fine))
        assert np.max(np.abs(values - expected)) < 1e-12
        print(f"hierarchy_beta={beta:g} coarse_eig={coarse_value:.8f} fine_eig={fine_value:.8f} above_half={sum(values>.5)}")
    # B.34: same-space symmetric perturbation, gap and projector estimates.
    base = target @ target.T
    perturbation = rng.normal(size=base.shape)
    perturbation = (perturbation + perturbation.T) / 2
    perturbation *= .1 / np.linalg.norm(perturbation, 2)
    values, vectors = np.linalg.eigh(base + perturbation)
    assert values[-3] - values[-4] >= 1 - 2*.1 - 1e-12
    assert np.linalg.norm(vectors[:, -3:] @ vectors[:, -3:].T - base, 2) <= .1/.9 + 1e-12
    for n_sites in [10, 50, 200]:
        for probability in [.01, .2, .5, .9]:
            tail = 0.
            for count in range(math.floor(n_sites*probability/2)+1):
                log_mass = (math.lgamma(n_sites+1)-math.lgamma(count+1)-math.lgamma(n_sites-count+1)
                            + count*np.log(probability)+(n_sites-count)*np.log1p(-probability))
                tail += np.exp(log_mass)
            assert tail <= np.exp(-n_sites*probability/8) + 1e-14


def main():
    check_cluster_bounds()
    check_localization_and_certificate()
    check_loading_and_sparse_baselines()
    check_diagnostics_hierarchy_sampling()
    print("all B2 checks passed")


if __name__ == "__main__":
    main()
