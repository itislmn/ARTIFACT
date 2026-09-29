"""
doa_methods.py — classical multi-snapshot DOA estimation, on your REAL array.

ARRAY GEOMETRY, not assumed - derived from your own antGeometryCfg line:
    antGeometryCfg 1 0 1 1 1 2 1 3 0 2 0 3 0 4 0 5 1 4 1 5 1 6 1 7 1 8 1 9 1 10 1 11 0.5 0.8

Parsing that line's 16 (row, col) pairs shows row=1 (the "azimuth" row) has
columns 0,1,2,...,11 - a perfect 12-element Uniform Linear Array (ULA) at
half-wavelength spacing. That matches your profile's own name:
"3Azim, 1Elev Tx" - 3 azimuth TX x 4 RX = 12 unique azimuth positions. This
module uses exactly those 12 virtual channels for azimuth DOA - real
geometry from your real hardware, not a toy assumption.

Every function takes `snapshots`: a (12, n_snapshots) complex array - one
column per independent look at the scene (one column per chirp, typically).
"""
import numpy as np

N_ELEMENTS = 12          # your real azimuth virtual array size
ELEMENT_SPACING = 0.5    # half-wavelength spacing, standard TI convention
AZIMUTH_CHANNEL_INDICES = [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15]
# ^ these are the cube's virtual-channel indices whose (row,col) = (1, 0..11)
#   in geometric column order 0..11 - see the parse above. If you ever
#   change antGeometryCfg, re-derive this list, don't assume it.


def steering_vector(angle_deg, n=N_ELEMENTS, spacing=ELEMENT_SPACING):
    """The array's expected phase pattern for a signal arriving from angle_deg.
    This is the physical model every DOA method below compares real data
    against."""
    theta = np.radians(angle_deg)
    element_idx = np.arange(n)
    phase = 2 * np.pi * spacing * element_idx * np.sin(theta)
    return np.exp(1j * phase)


def covariance_matrix(snapshots):
    """R = (1/K) * sum over snapshots of x @ x^H - the standard sample
    covariance every classical DOA method is built on."""
    K = snapshots.shape[1]
    return (snapshots @ snapshots.conj().T) / K


def bartlett(snapshots, angles_deg=np.linspace(-90, 90, 361)):
    """Conventional (Fourier) beamforming: just point a virtual beam in
    every direction and measure how much power arrives from there. The
    simplest possible DOA method, and a useful sanity baseline - if fancier
    methods disagree wildly with this, be suspicious of the fancier one."""
    R = covariance_matrix(snapshots)
    spectrum = np.array([
        np.real(steering_vector(a).conj() @ R @ steering_vector(a))
        for a in angles_deg
    ])
    return angles_deg, spectrum


def capon(snapshots, angles_deg=np.linspace(-90, 90, 361), diag_loading=1e-3):
    """Capon / MVDR: instead of just pointing a beam, actively try to
    MINIMIZE interference from every other direction while keeping the
    looked-at direction at unit gain. Sharper peaks than Bartlett,
    especially with multiple targets."""
    R = covariance_matrix(snapshots)
    R = R + diag_loading * np.eye(R.shape[0])   # regularization - real
    R_inv = np.linalg.inv(R)                     # data is noisy; this keeps
    spectrum = []                                # the inverse well-behaved
    for a in angles_deg:
        sv = steering_vector(a)
        denom = np.real(sv.conj() @ R_inv @ sv)
        spectrum.append(1.0 / denom)
    return angles_deg, np.array(spectrum)


def music(snapshots, n_targets, angles_deg=np.linspace(-90, 90, 361)):
    """MUSIC: split the covariance matrix's eigenvectors into a 'signal
    subspace' (the n_targets strongest) and a 'noise subspace' (the rest).
    A true target direction's steering vector is, in theory, PERFECTLY
    orthogonal to the noise subspace - so 1/(closeness to noise subspace)
    spikes exactly at real target angles. Needs to be told how many
    targets to look for in advance."""
    R = covariance_matrix(snapshots)
    eigvals, eigvecs = np.linalg.eigh(R)
    order = np.argsort(eigvals)[::-1]            # descending
    noise_subspace = eigvecs[:, order[n_targets:]]
    spectrum = []
    for a in angles_deg:
        sv = steering_vector(a)
        proj = noise_subspace.conj().T @ sv
        denom = np.real(proj.conj() @ proj)
        spectrum.append(1.0 / (denom + 1e-12))
    return angles_deg, np.array(spectrum)


def esprit(snapshots, n_targets, spacing=ELEMENT_SPACING):
    """ESPRIT: exploits the fact that a ULA has two identical, shifted
    sub-arrays (elements 0..10 and elements 1..11). The signal subspace
    of each sub-array is related by a simple rotation whose angle directly
    gives you the DOA - solved in closed form, no angle grid/search needed
    at all, unlike the three methods above."""
    R = covariance_matrix(snapshots)
    eigvals, eigvecs = np.linalg.eigh(R)
    order = np.argsort(eigvals)[::-1]
    Es = eigvecs[:, order[:n_targets]]            # signal subspace
    Es1, Es2 = Es[:-1, :], Es[1:, :]               # the two shifted sub-arrays
    # least-squares solution for the rotation operator between them
    phi = np.linalg.pinv(Es1) @ Es2
    eigvals_phi = np.linalg.eigvals(phi)
    angles = np.degrees(np.arcsin(
        np.clip(np.angle(eigvals_phi) / (2 * np.pi * spacing), -1, 1)))
    return np.sort(angles)


def find_peaks(angles_deg, spectrum, n_peaks):
    """Simple local-maximum peak picker for the spectral methods (Bartlett/
    Capon/MUSIC), used to turn a continuous spectrum into n_peaks angle
    estimates, comparable to ESPRIT's direct output."""
    s = 20 * np.log10(spectrum / spectrum.max() + 1e-12)
    is_peak = (s[1:-1] > s[:-2]) & (s[1:-1] > s[2:])
    peak_idx = np.where(is_peak)[0] + 1
    peak_idx = peak_idx[np.argsort(s[peak_idx])[::-1]][:n_peaks]
    return np.sort(angles_deg[peak_idx])
