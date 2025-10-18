# %%
"""
Brain Connectome ML Pipeline (Starter)
-------------------------------------
This script wires up an end‑to‑end workflow for:
  (1) Prediction tasks:
        - Sex (male vs. female)
        - High‑math vs. normal (via CCI threshold)
        - Creative vs. normal (via CAQ threshold)
  (2) Feature selection / invention (edges + graph features)
  (3) Visual analytics (group means and difference maps)

USAGE
-----
1) Put your individual subject .mat files in a folder, e.g. "/mnt/data/mat_subjects"
   - Each .mat should correspond to ONE subject, ideally with a filename that contains the URSI.
   - Each .mat should contain an adjacency matrix, common keys:
        'A', 'adj', 'adjacency', or a sparse triplet like 'i','j','w'.

2) Adjust CONFIG below: MAT_DIR and MAT_GLOB/GROUP, and any key names if needed.

3) Run this cell. It will:
   - Load Excel metadata (metainfo.xlsx)
   - Discover .mat files, map them to URSI
   - Build features (edge vector + basic graph metrics)
   - Prepare labels (Sex, HighMath, Creative)
   - (If enough data exists) do a quick cross‑validated baseline for each task

NOTES
-----
- The code is defensive: it will gracefully skip steps if .mat files are not present yet.
- All functions have clear docstrings for you to adapt.
"""

import os
import re
import json
import math
import glob
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union

from scipy.io import loadmat
from scipy import sparse
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

from IPython.display import display

def display_dataframe_to_user(title, df):
    print(f"\n{title}\n")
    display(df)



EXCEL_PATH = "metainfo.csv"

# Point this to your actual folder of .mat files
MAT_DIR = "mat_subjects/"   # <-- change if needed
MAT_GLOB = "*.mat"                   # pattern for .mat files

# Keys to try inside .mat files for adjacency matrix
ADJ_KEYS = ["A", "adj", "adjacency", "connectivity", "W"]

# Optional triplet keys (i,j,w) for sparse edge lists (1-indexed expected)
TRIPLET_KEYS = [("i", "j", "w"), ("row", "col", "val")]

# Thresholds (tweak to your dataset distribution)
CCI_HIGH_MATH_THRESHOLD = 100.0   # example; adjust per your study definition
CAQ_CREATIVE_THRESHOLD   = 20.0   # example; adjust per your study definition


# ---------------------------
# HELPERS
# ---------------------------
def normalize_ursi(raw: str) -> str:
    """
    Normalize URSI strings so they can be matched to filenames.
    e.g., "M87102217" -> "m87102217"
    """
    if pd.isna(raw):
        return ""
    return re.sub(r"[^A-Za-z0-9]+", "", str(raw)).lower()


def find_adjacency_in_mat(mat: dict) -> Optional[np.ndarray]:
    """
    Try common conventions to extract the adjacency matrix from a .mat dict.
    Returns a dense numpy array if found, else None.
    """
    # 1) Direct full matrix keys
    for k in ADJ_KEYS:
        if k in mat:
            A = mat[k]
            if sparse.issparse(A):
                return A.toarray()
            A = np.array(A)
            if A.ndim == 2 and A.shape[0] == A.shape[1]:
                return A

    # 2) Triplet sparse representation
    for i_key, j_key, w_key in TRIPLET_KEYS:
        if i_key in mat and j_key in mat and w_key in mat:
            i = mat[i_key].ravel()
            j = mat[j_key].ravel()
            w = mat[w_key].ravel()
            # Many MATLAB triplets are 1-based
            i = i.astype(int) - 1
            j = j.astype(int) - 1
            n = int(max(i.max(), j.max()) + 1)
            A = sparse.coo_matrix((w, (i, j)), shape=(n, n)).tocsr()
            # Assume undirected; symmetrize if needed
            A = A.maximum(A.T)
            return A.toarray()

    # 3) Heuristic: first square 2D array
    for k, v in mat.items():
        if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[0] == v.shape[1]:
            return v

    return None


def vectorize_upper_triangle(A: np.ndarray, k: int = 1) -> np.ndarray:
    """
    Vectorize the upper triangle (excluding diagonal by default) of adjacency A.
    k=1 excludes main diagonal.
    """
    idx = np.triu_indices_from(A, k=k)
    return A[idx]


def graph_features(A: np.ndarray) -> Dict[str, float]:
    """
    Compute simple graph features as 'feature invention' beyond raw edges.
    Uses weighted degree (node strength) and global measures.
    """
    # Symmetrize just in case
    A = 0.5 * (A + A.T)

    # Node strength (weighted degree)
    strength = A.sum(axis=1)

    # Global strength stats
    feats = {
        "strength_mean": float(np.mean(strength)),
        "strength_std": float(np.std(strength)),
        "strength_min": float(np.min(strength)),
        "strength_max": float(np.max(strength)),
    }

    # Density (fraction of nonzero edges in upper triangle)
    ut = vectorize_upper_triangle(A, k=1)
    density = float(np.count_nonzero(ut)) / float(ut.size)
    feats["density"] = density

    # Mean edge weight
    if ut.size > 0:
        feats["edge_mean"] = float(np.mean(ut))
        feats["edge_std"]  = float(np.std(ut))
    else:
        feats["edge_mean"] = 0.0
        feats["edge_std"]  = 0.0

    return feats


def ursi_from_filename(path: str) -> str:
    """
    Try to infer URSI from filename by extracting the longest alnum token.
    Customize if your naming scheme is different.
    """
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    # Common case: filename contains the URSI directly
    return normalize_ursi(stem)


def load_subject_adjacency(path: str) -> Optional[np.ndarray]:
    """
    Load a single subject .mat and return adjacency matrix if possible.
    """
    try:
        mat = loadmat(path)
        A = find_adjacency_in_mat(mat)
        return A
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
        return None


def build_feature_table(excel_df: pd.DataFrame, mat_paths: List[str]) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Match .mat files to Excel rows via URSI and create a feature table.
    Returns:
      - features_df: rows=subjects, columns=[edge_* ... graph_* ... labels]
      - adj_map: dict URSI -> adjacency matrix (for later visualizations)
    """
    # Normalize URSI in Excel
    excel_df = excel_df.copy()
    excel_df["URSI_norm"] = excel_df["URSI"].apply(normalize_ursi)

    # Map discovered mats
    path_map = {ursi_from_filename(p): p for p in mat_paths}

    records = []
    adj_map: Dict[str, np.ndarray] = {}

    # Determine N from first successful .mat
    N = None

    for _, row in excel_df.iterrows():
        ursi = row["URSI"]
        ukey = row["URSI_norm"]
        mat_path = path_map.get(ukey)
        if not mat_path:
            continue  # no matching .mat for this subject

        A = load_subject_adjacency(mat_path)
        if A is None:
            continue

        if N is None:
            N = A.shape[0]

        # Edge vector (upper triangle without diagonal)
        edge_vec = vectorize_upper_triangle(A, k=1)

        # Graph feature dict
        gfeats = graph_features(A)

        rec = {
            "URSI": ursi,
            "URSI_norm": ukey,
            # Labels
            "Sex": row.get("Sex", np.nan),
            "CCI": row.get("CCI", np.nan),
            "CAQ": row.get("CAQ", np.nan),
            "FSIQ": row.get("FSIQ", np.nan),
            "Age": row.get("Age", np.nan),
            "Subject_type": row.get("Subject_type", np.nan),
        }
        # Add graph features
        for k, v in gfeats.items():
            rec[f"graph_{k}"] = v

        # Add edge features with systematic names
        rec.update({f"edge_{i}": val for i, val in enumerate(edge_vec)})

        records.append(rec)
        adj_map[ukey] = A

    features_df = pd.DataFrame.from_records(records)
    return features_df, adj_map


def make_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary labels:
      - Sex: assumes 0=female,1=male (verify your coding!)
      - HighMath: 1 if CCI >= threshold else 0
      - Creative: 1 if CAQ >= threshold else 0
    """
    out = df.copy()
    out["Label_Sex"] = out["Sex"].astype(int)

    out["Label_HighMath"] = (out["CCI"] >= CCI_HIGH_MATH_THRESHOLD).astype(int)
    out["Label_Creative"] = (out["CAQ"] >= CAQ_CREATIVE_THRESHOLD).astype(int)
    return out


from typing import Any

def quick_cv_classify(X: np.ndarray, y: np.ndarray, model_name: str, k: int = 5) -> Dict[str, Union[float, str]]:
    """
    Run a quick k‑fold stratified CV with a simple pipeline:
      - Standardize
      - SelectKBest (ANOVA) limited to min(300, X.shape[1]) features
      - Logistic Regression (liblinear) OR RandomForest fallback
    Returns a metrics dict with mean accuracy/F1/AUC where applicable.
    """
    # Guard for tiny datasets
    if len(np.unique(y)) < 2 or X.shape[0] < k:
        return {"n": int(X.shape[0]), "note": "Not enough samples or only one class present."}

    kbest = min(300, X.shape[1]) if X.shape[1] > 10 else X.shape[1]

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False) if sparse.issparse(X) else StandardScaler()),
        ("sel", SelectKBest(score_func=f_classif, k=kbest)),
        ("clf", LogisticRegression(penalty="l1", solver="liblinear", max_iter=2000))
    ])

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    accs, f1s, aucs = [], [], []
    for train_idx, test_idx in skf.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        pipe.fit(Xtr, ytr)
        yp = pipe.predict(Xte)
        yps = None
        try:
            yps = pipe.predict_proba(Xte)[:, 1]
        except Exception:
            pass
        accs.append(accuracy_score(yte, yp))
        f1s.append(f1_score(yte, yp, zero_division=0))
        if yps is not None and len(np.unique(y)) == 2:
            try:
                aucs.append(roc_auc_score(yte, yps))
            except Exception:
                pass

    out = {
        "model": model_name,
        "n": int(X.shape[0]),
        "acc_mean": float(np.mean(accs)) if accs else None,
        "f1_mean": float(np.mean(f1s)) if f1s else None,
        "auc_mean": float(np.mean(aucs)) if aucs else None
    }
    return out


def visualize_group_means(adj_map: Dict[str, np.ndarray],
                          labels: pd.Series,
                          title_left: str,
                          title_right: str,
                          ursi_norm_series: pd.Series) -> None:
    """
    Visualize group mean adjacency matrices and their difference.
    - labels: pd.Series indexed by row of features_df with 0/1
    - adj_map: URSI_norm -> A
    """
    # Gather matrices by group
    group0, group1 = [], []
    for ukey, lab in zip(ursi_norm_series, labels):
        A = adj_map.get(ukey)
        if A is None:
            continue
        if lab == 0:
            group0.append(A)
        elif lab == 1:
            group1.append(A)

    if not group0 or not group1:
        print("[INFO] Not enough data to visualize both groups yet.")
        return

    mean0 = np.mean(group0, axis=0)
    mean1 = np.mean(group1, axis=0)
    diff  = mean1 - mean0

    # Plot each as a heatmap
    for mat, t in [(mean0, f"{title_left} (mean)"),
                   (mean1, f"{title_right} (mean)"),
                   (diff,  f"Difference (Right - Left)")]:

        plt.figure()
        plt.imshow(mat)
        plt.title(t)
        plt.colorbar()
        plt.xlabel("Region")
        plt.ylabel("Region")
        plt.show()


# ---------------------------
# LOAD EXCEL & DISCOVER .MAT
# ---------------------------
meta = pd.read_excel(EXCEL_PATH, sheet_name="Sheet1")

# Show a preview to the user in a nice table
display_dataframe_to_user("Subject Metadata (Sheet1)", meta.head(20))

mat_files = glob.glob('mat_subjects/*.mat')

summary_info = {
    "excel_rows": int(len(meta)),
    "mat_dir_exists": os.path.isdir(MAT_DIR),
    "mat_files_found": len(mat_files),
    "mat_dir": MAT_DIR,
    "hint": "Place your subject .mat files in MAT_DIR and ensure filenames include the URSI (e.g., M87102217.mat)."
}
