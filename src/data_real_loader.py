import os
import glob
import pandas as pd
import numpy as np

# Format cible: frame_id, ped_id, x, y
TARGET_COLS = ["frame_id", "ped_id", "x", "y"]

def read_any_traj_file(path: str) -> pd.DataFrame:
    """
    Lit un fichier trajectoire ETH/UCY quelle que soit la forme (csv/tsv/txt).
    Essaie de détecter automatiquement le séparateur et le nombre de colonnes.

    ETH/UCY existent souvent en:
    - tab/space separated
    - colonnes typiques: frame, ped, x, y (parfois + vx vy ou + autre)
    """
    # Essai séparateurs courants
    for sep in [",", "\t", r"\s+"]:
        try:
            df = pd.read_csv(path, sep=sep, header=None, engine="python")
            if df.shape[1] >= 4:
                break
        except Exception:
            df = None

    if df is None or df.shape[1] < 4:
        raise ValueError(f"Impossible de lire {path}: format non reconnu.")

    # Prend les 4 premières colonnes comme frame, ped, x, y (standard le plus courant)
    df = df.iloc[:, :4].copy()
    df.columns = TARGET_COLS

    # Types
    df["frame_id"] = pd.to_numeric(df["frame_id"], errors="coerce").astype("Int64")
    df["ped_id"] = pd.to_numeric(df["ped_id"], errors="coerce").astype("Int64")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    df = df.dropna().copy()
    df["frame_id"] = df["frame_id"].astype(int)
    df["ped_id"] = df["ped_id"].astype(int)

    return df


def basic_cleaning(df: pd.DataFrame, min_track_len: int = 20) -> pd.DataFrame:
    """
    - trie par (ped_id, frame_id)
    - supprime doublons (ped, frame)
    - garde uniquement les trajectoires assez longues
    """
    df = df.sort_values(["ped_id", "frame_id"]).copy()
    df = df.drop_duplicates(subset=["ped_id", "frame_id"], keep="first")

    # Filtre trajectoires trop courtes
    lengths = df.groupby("ped_id")["frame_id"].count()
    keep_ids = lengths[lengths >= min_track_len].index
    df = df[df["ped_id"].isin(keep_ids)].copy()

    return df


def add_scene_id(df: pd.DataFrame, scene_name: str) -> pd.DataFrame:
    df = df.copy()
    df.insert(0, "scene", scene_name)
    return df


def process_folder(raw_folder: str, out_path: str, min_track_len: int = 20):
    """
    Prend tous les fichiers all_raw_data d'un dossier (une scène ou un groupe de scènes),
    les concatène, nettoie, et écrit un CSV standardisé.
    """
    patterns = ["*.txt", "*.csv", "*.tsv", "*.dat"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(raw_folder, p)))

    if not files:
        raise FileNotFoundError(f"Aucun fichier trouvé dans {raw_folder}")

    all_dfs = []
    for f in sorted(files):
        scene = os.path.splitext(os.path.basename(f))[0]
        df = read_any_traj_file(f)
        df = basic_cleaning(df, min_track_len=min_track_len)
        df = add_scene_id(df, scene)
        all_dfs.append(df)

    out = pd.concat(all_dfs, axis=0, ignore_index=True)
    out = out.sort_values(["scene", "ped_id", "frame_id"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[OK] Saved: {out_path} | rows={len(out)} | scenes={out['scene'].nunique()}")


if __name__ == "__main__":
    # Exemple:
    # python src/data_real_loader.py
    RAW = "data_real/all_raw_data"                # racine
    OUT = "data_real/processed"          # sortie

    # Tu peux lancer sur un dossier ETH ou UCY
    # Exemple: data_real/all_raw_data/eth/  et data_real/all_raw_data/ucy/
    for dataset_name in ["eth", "ucy"]:
        in_dir = os.path.join(RAW, dataset_name)
        if os.path.isdir(in_dir):
            out_file = os.path.join(OUT, f"{dataset_name}_all_scenes.csv")
            process_folder(in_dir, out_file, min_track_len=20)
        else:
            print(f"[SKIP] folder not found: {in_dir}")
