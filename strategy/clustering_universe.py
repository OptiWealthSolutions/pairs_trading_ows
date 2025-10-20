#Asset Selection - Subtractive Clustering in Orthogonal Eigenspace
#For this we employ a combination of Principle Component Analysis (PCA) and subtractive clustering.

#Let RT×M be the daily bank excess returns (henceforth the term “return” describes bank-excess return) of M assets for
# T days, and define rm to be the daily returns of asset m. Define ym =
# rm−µm/om
# and YT×M = [y1..y2...yM] ,where µm = E(rm)
# and σm = Var(rm).

import numpy as np
from numpy.linalg import svd, norm
from sklearn.preprocessing import StandardScaler

# --- 1) préparation des données ---
# R : array shape (T, M) rendements journaliers (bank-excess returns).
# Exemple si tu veux tester : uncomment below
T, M = 500, 50
np.random.seed(0)
R = np.random.randn(T, M) * 0.02  # rendements simulés

def prepare_Y(R):
    """Normalise chaque colonne (actif) -> Y (T x M)."""
    scaler = StandardScaler(with_mean=True, with_std=True)
    Y = scaler.fit_transform(R)  # centre & met à l'échelle (std = 1)
    return Y, scaler.mean_, scaler.scale_

# --- 2) SVD / PCA ---
def compute_svd(Y, K):
    """
    Renvoie U_K, Sigma_K, Vt_K.
    Y approx = U_K @ Sigma_K @ Vt_K
    """
    U, s, Vt = svd(Y, full_matrices=False)  # s: singular values, len = min(T,M)
    U_k = U[:, :K]
    s_k = s[:K]
    Vt_k = Vt[:K, :]
    Sigma_k = np.diag(s_k)
    return U_k, Sigma_k, Vt_k, s_k

# --- 3) coordonnées des actifs dans l'espace factoriel ---
def asset_coords_from_vt(Vt_k):
    """
    Vt_k shape (K, M). Colonnes de beta sont les coordonnées b_m.
    On retourne X shape (M, K) avec chaque ligne = b_m^T.
    """
    return Vt_k.T  # (M, K)

# --- 4) pondération ellipsoïdale selon singular values ---
def scale_for_ellipsoid(X, s_k):
    """
    s_k: singular values (length K).
    Poids par dimension = 1 / s_k^2 (comme dans ton texte).
    On normalise les poids pour garder l'échelle numérique stable.
    On applique sqrt(weight) aux colonnes pour transformer l'ellipsoïde en sphère.
    """
    w = 1.0 / (s_k**2 + 1e-12)             # éviter div0
    w = w / np.sum(w)                     # normaliser somme = 1 (optionnel)
    scale = np.sqrt(w)                    # facteur multiplicatif par dimension
    Xs = X * scale[np.newaxis, :]         # broadcasting sur M lignes
    return Xs, scale, w

# --- 5) subtractive clustering (implémentation standard) ---
def subtractive_clustering(Xs, ra=0.5, rb=None, alpha=1.0, max_centers=20, density_threshold=1e-4):
    """
    Xs: (M, K) données (après scaling).
    ra: neighborhood radius (en unités de Xs). Valeur par défaut 0.5 (à ajuster).
    rb: radius for density reduction, par défaut = 1.5 * ra.
    alpha: multiplicative for density reduction (classiquement = 1).
    Retour: centers (list of indices), density values
    Référence: Chiu (1994) subtractive clustering.
    """
    if rb is None:
        rb = 1.5 * ra
    M, K = Xs.shape
    # calcul des densités
    # density_i = sum_j exp(-4 * ||x_i - x_j||^2 / ra^2)
    # facteur -4 suit la convention usuelle mais tu peux ajuster
    sqd = np.sum((Xs[:, np.newaxis, :] - Xs[np.newaxis, :, :])**2, axis=2)  # (M,M)
    density = np.sum(np.exp(-4.0 * sqd / (ra**2)), axis=1)
    centers = []
    densities = []
    density_copy = density.copy()
    for _ in range(max_centers):
        idx = np.argmax(density_copy)
        max_d = density_copy[idx]
        if max_d < density_threshold:
            break
        centers.append(idx)
        densities.append(max_d)
        # reduction step: density_copy_j = density_copy_j - max_d * exp(-4*||x_j-x_idx||^2 / rb^2)
        dist2_to_center = sqd[idx]
        reduction = max_d * np.exp(-4.0 * dist2_to_center / (rb**2))
        density_copy = density_copy - reduction
        density_copy = np.maximum(density_copy, 0.0)
    return centers, np.array(densities), density

# --- 6) assigner chaque actif au centre le plus proche (dans espace original non-scaled ou scaled selon choix) ---
def assign_clusters(Xs, centers):
    if len(centers) == 0:
        return np.array([-1]*Xs.shape[0])
    cent_coords = Xs[centers]  # (C, K)
    # distance de chaque point au centre
    d2 = np.sum((Xs[:, np.newaxis, :] - cent_coords[np.newaxis, :, :])**2, axis=2)
    labels = np.argmin(d2, axis=1)
    return labels

# --- Pipeline complet ---
def pca_subtractive_pipeline(R, K=3, ra=None, rb=None, max_centers=20):
    """
    R: (T, M) rendements
    K: nb de composantes à garder
    ra: neighborhood radius after scaling. Si None on calcule une valeur par défaut.
    """
    Y, means, scales = prepare_Y(R)                 # (T, M)
    U_k, Sigma_k, Vt_k, s_k = compute_svd(Y, K)
    X = asset_coords_from_vt(Vt_k)                  # (M, K)
    Xs, scale_vec, raw_w = scale_for_ellipsoid(X, s_k)  # transforme en espace sphérique
    
    # heuristique légère pour ra si non fournie : fraction de la moyenne des distances interpoints
    if ra is None:
        dists = np.sqrt(np.sum((Xs[:, np.newaxis, :] - Xs[np.newaxis, :, :])**2, axis=2))
        mean_dist = np.mean(dists[np.triu_indices(Xs.shape[0], k=1)])
        ra = 0.5 * mean_dist  # réglage raisonnable par défaut
    
    centers, densities, density_all = subtractive_clustering(Xs, ra=ra, rb=rb, max_centers=max_centers)
    labels = assign_clusters(Xs, centers)
    
    results = {
        "Y": Y,
        "s_k": s_k,
        "X": X,
        "Xs": Xs,
        "scale_vec": scale_vec,
        "centers_idx": centers,
        "centers_coords": X[centers] if centers else np.array([]),
        "densities": densities,
        "labels": labels,
        "ra": ra
    }
    return results

# --- Exemple d'utilisation minimal ---
if __name__ == "__main__":
    # Test rapide avec données simulées
    T, M = 500, 50
    np.random.seed(42)
    R = np.random.randn(T, M) * 0.02
    # introduisons 3 groupes pour tester
    for i in range(M):
        R[:, i] += 0.005 * (i // 16)  # shift group effect
    out = pca_subtractive_pipeline(R, K=3, max_centers=10)
    print("Found centers (indices):", out["centers_idx"])
    print("Cluster counts:", np.bincount(out["labels"]))


    # 	Identifier des actifs “représentatifs” par cluster.
	# •	Étudier la corrélation intra-cluster (devrait être forte).
	# •	Utiliser ces groupes pour diversification, sélection de pairs, ou réduction d’univers.