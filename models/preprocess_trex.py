import numpy as np

def load_obj(filename):
    """Minimal OBJ loader (vertices + faces only)."""
    verts = []
    faces = []
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                # Face indices are 1-based in OBJ
                face = [int(p.split("/")[0]) - 1 for p in parts]
                faces.append(face)
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)

def preprocess_obj(verts, faces, scale=0.05):
    """
    Normalize OBJ model:
    - Center at origin
    - Scale relative to marker size (in meters)
    """
    # --- Center model ---
    center = verts.mean(axis=0)
    verts -= center

    # --- Scale model ---
    max_dim = np.max(verts.max(axis=0) - verts.min(axis=0))
    scale_factor = scale / max_dim
    verts *= scale_factor

    print(f"✅ Preprocessed Trex: scale_factor={scale_factor:.5f}, centered at origin")
    return verts, faces

if __name__ == "__main__":
    # --- Parameters ---
    obj_file = "./trex_model.obj"     # make sure this file exists
    out_file = "trex_preprocess.npz"
    marker_size = 0.05              # Marker size in meters (adjust if needed)

    # --- Run Preprocess ---
    verts, faces = load_obj(obj_file)
    verts, faces = preprocess_obj(verts, faces, scale=marker_size)

    # Save preprocessed model
    np.savez(out_file, verts=verts, faces=faces)
    print(f"💾 Saved preprocessed model to {out_file}")
