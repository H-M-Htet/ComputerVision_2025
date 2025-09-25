import numpy as np

def load_obj(path):
    verts = []
    faces = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):  # vertex
                _, x, y, z = line.split()
                verts.append([float(x), float(y), float(z)])
            elif line.startswith("f "):  # face
                parts = line.split()[1:]
                face = [int(p.split("/")[0]) - 1 for p in parts]
                faces.append(face)
    return np.array(verts, dtype=np.float32), faces

def normalize_model(verts, target_height=0.1):
    # Bounding box
    min_vals = verts.min(axis=0)
    max_vals = verts.max(axis=0)
    size = max_vals - min_vals
    height = size[1]  # assuming Y is "up" in your model

    # Shift so feet touch ground (min_y = 0)
    verts[:, 1] -= min_vals[1]

    # Center X and Z around 0
    verts[:, 0] -= (min_vals[0] + size[0] / 2)
    verts[:, 2] -= (min_vals[2] + size[2] / 2)

    # Scale to target height (meters)
    scale = target_height / height
    verts *= scale

    return verts, scale

if __name__ == "__main__":
    verts, faces = load_obj("models/trex_model.obj")
    print(f"Original vertices: {verts.shape}, faces: {len(faces)}")

    verts_norm, scale = normalize_model(verts, target_height=0.1)
    print(f"Scaled with factor {scale:.6f}")
    print(f"Bounding box after normalization:")
    print(f"  min={verts_norm.min(axis=0)}, max={verts_norm.max(axis=0)}")

    # Save preprocessed version for AR
    np.savez("trex_preprocessed.npz", verts=verts_norm, faces=np.array(faces, dtype=object))
    print("✅ Saved normalized model -> trex_preprocessed.npz")
