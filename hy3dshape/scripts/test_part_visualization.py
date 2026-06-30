
"""
test_part_visualization.py

WHAT THIS DOES:
  1. Loads a mesh (your prior)
  2. Encodes it to latents via ShapeVAE  [1, 512, 64]
  3. For each token chunk: zeros it out → decodes → compares vertices
     to baseline → measures & visualizes what geometry was affected
  4. Produces:
       - ablation_summary.png   : heatmap of which chunks affect which
                                  spatial regions (top/mid/bottom)
       - chunk_NNN_NNN.obj      : each ablated mesh for manual inspection
       - pattern_report.txt     : tells you Pattern A/B/C result
                                  and suggested PART_TOKEN_INDICES

REQUIREMENTS (all in hy3dshape env):
  trimesh, torch, numpy, matplotlib
  ShapeVAE, SharpEdgeSurfaceLoader from hy3dshape

USAGE:
  python test_part_visualization.py \
      --mesh   path/to/prior.obj \
      --vae    tencent/Hunyuan3D-2mini \
      --subfolder hunyuan3d-vae-v2-mini-withencoder \
      --out    ./part_vis_outputs \
      --chunk  64
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')   # headless — works on SLURM
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import trimesh

sys.path.insert(0, './hy3dshape')

# ── Argument parsing ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--mesh',       required=True,  help='Path to prior .obj/.glb')
parser.add_argument('--vae',        default='tencent/Hunyuan3D-2mini')
parser.add_argument('--subfolder',  default='hunyuan3d-vae-v2-mini-withencoder')
parser.add_argument('--out',        default='./part_vis_outputs')
parser.add_argument('--chunk',      type=int, default=64,
                    help='Token chunk size for ablation (64 = 8 chunks over 512)')
parser.add_argument('--resolution', type=int, default=256,
                    help='Marching cubes resolution for decode')
parser.add_argument('--no-export-obj', action='store_true',
                    help='Skip .obj export (faster, just produces plots)')
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

# ── Imports from hy3dshape ────────────────────────────────────────────────
try:
    from hy3dshape.models.autoencoders import ShapeVAE
    from hy3dshape.surface_loaders import SharpEdgeSurfaceLoader
    from hy3dshape.pipelines import export_to_trimesh
except ImportError as e:
    print(f"[ERROR] hy3dshape import failed: {e}")
    print("Make sure you're running inside the hy3dshape conda/venv environment.")
    sys.exit(1)

# ── Load VAE ──────────────────────────────────────────────────────────────
print(f"[1/5] Loading ShapeVAE from {args.vae} ...")
vae = ShapeVAE.from_pretrained(
    args.vae,
    subfolder=args.subfolder,
    use_safetensors=False,
    variant='fp16',
).to('cuda').eval()
print(f"      latent_shape = {vae.latent_shape}")   # expect (512, 64)

LATENT_TOKENS = vae.latent_shape[0]   # 512
LATENT_DIM    = vae.latent_shape[1]   # 64

# ── Load mesh + encode ────────────────────────────────────────────────────
print(f"[2/5] Loading mesh: {args.mesh}")
mesh_orig = trimesh.load(args.mesh, force='mesh')
print(f"      vertices={len(mesh_orig.vertices)}  faces={len(mesh_orig.faces)}")

loader = SharpEdgeSurfaceLoader(
    num_sharp_points=0,
    num_uniform_points=81920,
)
surface = loader(mesh_orig).to('cuda', dtype=torch.float16)
print(f"      surface tensor: {surface.shape}")

print("[3/5] Encoding to latent ...")
with torch.no_grad():
    z = vae.encode(surface)   # [1, 512, 64]
print(f"      z.shape = {z.shape}   dtype={z.dtype}")

# ── Introspect VAE ────────────────────────────────────────────────────────
print("\n      [VAE introspection]")
print(f"        type         : {type(vae).__name__}")
print(f"        latent_shape : {vae.latent_shape}")
decode_methods = [m for m in dir(vae)
                  if not m.startswith('_')
                  and ('decode' in m.lower() or 'mesh' in m.lower()
                       or 'forward' in m.lower())]
print(f"        decode-like methods: {decode_methods}")
if hasattr(vae, 'decoder'):
    first_linears = [(n, m) for n, m in vae.decoder.named_modules()
                     if isinstance(m, torch.nn.Linear)]
    if first_linears:
        n0, m0 = first_linears[0]
        print(f"        decoder first Linear '{n0}': "
              f"in={m0.in_features} out={m0.out_features}")
        print(f"        → decoder expects latent dim = {m0.in_features}")
print()

# ── Probe correct decode path once ───────────────────────────────────────
def _probe_decode_path():
    """
    ShapeVAE has two possible decode paths depending on version:
      Path A: vae.decode(z)            → decoded_latent  → latents2mesh(decoded)
      Path B: vae.latents2mesh(z)      → works directly  (older API)
      Path C: vae(z)                   → mesh directly

    We probe all three and remember which works.
    """
    test_z = z.clone()
    
    # Path A: decode first, then latents2mesh
    try:
        with torch.no_grad():
            decoded = vae.decode(test_z)        # [1, 512, 1024] or similar
            out = vae.latents2mesh(
                decoded,
                bounds=1.01, mc_level=0.0,
                num_chunks=20000,
                octree_resolution=args.resolution,
                mc_algo='mc', enable_pbar=False,
            )
        result = export_to_trimesh(out)
        if isinstance(result, list): result = result[0]
        if result is not None and len(result.vertices) > 0:
            print("      [probe] decode path: A  (vae.decode → latents2mesh)")
            return 'A'
    except Exception as e:
        print(f"      [probe] path A failed: {e}")

    # Path B: latents2mesh directly (some VAE versions project internally)
    try:
        with torch.no_grad():
            out = vae.latents2mesh(
                test_z,
                bounds=1.01, mc_level=0.0,
                num_chunks=20000,
                octree_resolution=args.resolution,
                mc_algo='mc', enable_pbar=False,
            )
        result = export_to_trimesh(out)
        if isinstance(result, list): result = result[0]
        if result is not None and len(result.vertices) > 0:
            print("      [probe] decode path: B  (latents2mesh directly)")
            return 'B'
    except Exception as e:
        print(f"      [probe] path B failed: {e}")

    # Path C: try vae.forward / __call__
    try:
        with torch.no_grad():
            out = vae(test_z,
                      bounds=1.01, mc_level=0.0,
                      num_chunks=20000,
                      octree_resolution=args.resolution,
                      mc_algo='mc', enable_pbar=False)
        result = export_to_trimesh(out)
        if isinstance(result, list): result = result[0]
        if result is not None and len(result.vertices) > 0:
            print("      [probe] decode path: C  (vae.__call__)")
            return 'C'
    except Exception as e:
        print(f"      [probe] path C failed: {e}")

    # Path D: inspect vae for any decode-like method
    decode_candidates = [m for m in dir(vae)
                         if 'decode' in m.lower() or 'mesh' in m.lower()]
    print(f"      [probe] FAILED all paths. VAE decode methods found: {decode_candidates}")
    print(f"      [probe] VAE type: {type(vae)}")
    print(f"      [probe] z.shape: {test_z.shape}  z.dtype: {test_z.dtype}")
    return None

DECODE_PATH = _probe_decode_path()
if DECODE_PATH is None:
    print("[ERROR] Could not find working VAE decode path.")
    print("        Please check vae decode API and update decode_to_mesh() manually.")
    sys.exit(1)

# ── Decode helper ─────────────────────────────────────────────────────────
def decode_to_mesh(latent: torch.Tensor) -> trimesh.Trimesh | None:
    try:
        with torch.no_grad():
            if DECODE_PATH == 'A':
                decoded = vae.decode(latent)
                outputs = vae.latents2mesh(
                    decoded,
                    bounds=1.01, mc_level=0.0,
                    num_chunks=20000,
                    octree_resolution=args.resolution,
                    mc_algo='mc', enable_pbar=False,
                )
            elif DECODE_PATH == 'B':
                outputs = vae.latents2mesh(
                    latent,
                    bounds=1.01, mc_level=0.0,
                    num_chunks=20000,
                    octree_resolution=args.resolution,
                    mc_algo='mc', enable_pbar=False,
                )
            else:  # C
                outputs = vae(
                    latent,
                    bounds=1.01, mc_level=0.0,
                    num_chunks=20000,
                    octree_resolution=args.resolution,
                    mc_algo='mc', enable_pbar=False,
                )
        result = export_to_trimesh(outputs)
        if isinstance(result, list):
            result = result[0]
        return result
    except Exception as e:
        print(f"      [WARN] decode failed: {e}")
        return None

# ── Baseline decode ───────────────────────────────────────────────────────
print("[4/5] Decoding baseline mesh ...")
baseline_mesh = decode_to_mesh(z)
if baseline_mesh is None:
    print("[ERROR] Baseline decode failed — check VAE + mesh.")
    sys.exit(1)

baseline_verts = np.array(baseline_mesh.vertices)   # [V, 3]
print(f"      baseline: {len(baseline_verts)} vertices")

if not args.no_export_obj:
    baseline_mesh.export(os.path.join(args.out, "00_baseline.obj"))
    print(f"      saved 00_baseline.obj")

# Spatial bounds of baseline mesh
v = baseline_verts
bounds = {
    'x': (v[:, 0].min(), v[:, 0].max()),
    'y': (v[:, 1].min(), v[:, 1].max()),   # height axis
    'z': (v[:, 2].min(), v[:, 2].max()),
}
print(f"      mesh bounds  X:{bounds['x']}  Y:{bounds['y']}  Z:{bounds['z']}")

# ── Spatial region definitions ────────────────────────────────────────────
y_min, y_max = bounds['y']
y_range = y_max - y_min

SPATIAL_REGIONS = {
    'top_third':    (y_min + y_range * 0.66,  y_max),
    'middle_third': (y_min + y_range * 0.33,  y_min + y_range * 0.66),
    'bottom_third': (y_min,                   y_min + y_range * 0.33),
    'top_half':     (y_min + y_range * 0.50,  y_max),
    'bottom_half':  (y_min,                   y_min + y_range * 0.50),
}

def region_mask(verts: np.ndarray, y_lo: float, y_hi: float) -> np.ndarray:
    return (verts[:, 1] >= y_lo) & (verts[:, 1] < y_hi)

# ── Ablation loop ─────────────────────────────────────────────────────────
print(f"[5/5] Ablation: chunk_size={args.chunk}  "
      f"n_chunks={LATENT_TOKENS // args.chunk} ...")

CHUNK = args.chunk
n_chunks = LATENT_TOKENS // CHUNK

# Results table: rows=chunks, cols=spatial_regions
#   value = mean vertex displacement in that region when chunk is zeroed
displacement_table = np.zeros((n_chunks, len(SPATIAL_REGIONS)))
region_names = list(SPATIAL_REGIONS.keys())

chunk_results = []   # list of dicts for report

for ci in range(n_chunks):
    start = ci * CHUNK
    end   = start + CHUNK

    z_abl = z.clone()
    z_abl[:, start:end, :] = 0.0

    m = decode_to_mesh(z_abl)
    label = f"chunk_{start:03d}_{end:03d}"

    if m is None:
        print(f"  [{ci+1}/{n_chunks}] {label}: decode failed, skipping")
        chunk_results.append({'label': label, 'failed': True})
        continue

    abl_verts = np.array(m.vertices)

    # Export .obj for manual inspection
    if not args.no_export_obj:
        m.export(os.path.join(args.out, f"{label}.obj"))

    # Measure per-region displacement
    # Only valid if vertex count matches baseline
    row_displacements = {}
    if len(abl_verts) == len(baseline_verts):
        diff = np.linalg.norm(abl_verts - baseline_verts, axis=1)  # [V]
        for ri, (rname, (y_lo, y_hi)) in enumerate(SPATIAL_REGIONS.items()):
            mask = region_mask(baseline_verts, y_lo, y_hi)
            if mask.sum() > 0:
                d = diff[mask].mean()
            else:
                d = 0.0
            displacement_table[ci, ri] = d
            row_displacements[rname] = d
    else:
        # Vertex count changed — mesh topology broken, use bbox change as proxy
        bb_orig = baseline_verts.max(0) - baseline_verts.min(0)
        bb_abl  = abl_verts.max(0)  - abl_verts.min(0)
        proxy   = float(np.linalg.norm(bb_orig - bb_abl))
        displacement_table[ci, :] = proxy
        row_displacements = {r: proxy for r in region_names}

    # Summary line
    top_d    = row_displacements.get('top_third',    0)
    mid_d    = row_displacements.get('middle_third', 0)
    bot_d    = row_displacements.get('bottom_third', 0)
    dominant = max(row_displacements, key=row_displacements.get)

    print(f"  [{ci+1:2d}/{n_chunks}] {label}  "
          f"top={top_d:.4f}  mid={mid_d:.4f}  bot={bot_d:.4f}  "
          f"→ dominant: {dominant}")

    chunk_results.append({
        'label':        label,
        'start':        start,
        'end':          end,
        'failed':       False,
        'displacements': row_displacements,
        'dominant':     dominant,
    })

# ── Determine pattern ─────────────────────────────────────────────────────
valid = [r for r in chunk_results if not r.get('failed')]
if valid:
    all_d = displacement_table[
        [r['start'] // CHUNK for r in valid], :
    ]
    # Pattern A: clear spatial ordering — top chunks affect top, etc.
    # Pattern B: distributed — each chunk affects all regions roughly equally
    # Pattern C: global — all chunks have similar flat displacement profile
    per_chunk_max  = all_d.max(axis=1)
    per_chunk_std  = all_d.std(axis=1)
    spatial_std    = displacement_table.std(axis=0).mean()
    
    if spatial_std > 0.02 and per_chunk_std.mean() > 0.01:
        pattern = 'A'
        pattern_desc = (
            "SPATIALLY ORDERED — token ranges have clear geometric correspondence.\n"
            "→ You can define PART_TOKEN_INDICES directly from this ablation.\n"
            "→ PartStructureEncoder + per-part noise will work as designed."
        )
    elif per_chunk_std.mean() < 0.005:
        pattern = 'C'
        pattern_desc = (
            "GLOBAL — zeroing any chunk degrades the whole mesh equally.\n"
            "→ Tokens encode frequency/global features, not spatial regions.\n"
            "→ Recommendation: define parts on the MESH before VAE encoding.\n"
            "   Encode cavity submesh and shell submesh separately."
        )
    else:
        pattern = 'B'
        pattern_desc = (
            "DISTRIBUTED — tokens are entangled across spatial regions.\n"
            "→ No clean index ranges. Need learned part queries or\n"
            "   cluster tokens in latent space (k-means on z[0]).\n"
            "→ Fallback: encode cavity/shell as separate VAE passes."
        )
else:
    pattern = '?'
    pattern_desc = "Could not determine pattern — all decodes failed."

# ── Plot 1: Ablation heatmap ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle(
    f"ShapeVAE Latent Ablation  |  chunk_size={CHUNK}  |  Pattern: {pattern}",
    fontsize=14, fontweight='bold'
)

# Left: heatmap rows=chunks, cols=spatial regions
ax = axes[0]
im = ax.imshow(
    displacement_table,
    aspect='auto',
    cmap='hot',
    interpolation='nearest',
)
ax.set_title("Mean Vertex Displacement\n(darker = more affected)", fontsize=11)
ax.set_xlabel("Spatial Region")
ax.set_ylabel("Token Chunk (zeroed out)")
ax.set_xticks(range(len(region_names)))
ax.set_xticklabels(region_names, rotation=30, ha='right', fontsize=8)
ax.set_yticks(range(n_chunks))
ax.set_yticklabels(
    [f"{i*CHUNK}-{(i+1)*CHUNK}" for i in range(n_chunks)],
    fontsize=7
)
plt.colorbar(im, ax=ax, label='Displacement')

# Right: bar chart — dominant affected region per chunk
ax2 = axes[1]
colors_map = {
    'top_third':    '#e74c3c',
    'middle_third': '#f39c12',
    'bottom_third': '#2ecc71',
    'top_half':     '#c0392b',
    'bottom_half':  '#27ae60',
}
bar_colors = []
bar_heights = []
bar_labels  = []
for ci, res in enumerate(chunk_results):
    if res.get('failed'):
        bar_heights.append(0)
        bar_colors.append('#cccccc')
        bar_labels.append('failed')
    else:
        dom = res['dominant']
        bar_heights.append(res['displacements'].get(dom, 0))
        bar_colors.append(colors_map.get(dom, '#7f8c8d'))
        bar_labels.append(dom)

x_pos = range(len(chunk_results))
bars = ax2.bar(x_pos, bar_heights, color=bar_colors, edgecolor='white', linewidth=0.5)
ax2.set_title("Dominant Affected Region per Chunk", fontsize=11)
ax2.set_xlabel("Chunk index")
ax2.set_ylabel("Max displacement")
ax2.set_xticks(list(x_pos))
ax2.set_xticklabels(
    [f"{i*CHUNK}" for i in range(n_chunks)],
    rotation=45, fontsize=7
)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=c, label=r)
    for r, c in colors_map.items()
    if r in set(bar_labels)
]
ax2.legend(handles=legend_elements, fontsize=8, loc='upper right')

plt.tight_layout()
heatmap_path = os.path.join(args.out, "ablation_summary.png")
plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {heatmap_path}")

# ── Plot 2: Per-chunk spatial profile (line plot) ─────────────────────────
fig2, ax3 = plt.subplots(figsize=(14, 5))
region_colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6']
x = np.arange(n_chunks)
for ri, rname in enumerate(['top_third', 'middle_third', 'bottom_third']):
    ri_idx = region_names.index(rname)
    ax3.plot(
        x,
        displacement_table[:, ri_idx],
        marker='o', markersize=4,
        label=rname,
        color=region_colors[ri],
        linewidth=1.5,
    )
ax3.set_title(
    "Displacement Profile per Token Chunk\n"
    "Peaked lines = spatial correspondence  |  Flat lines = global encoding",
    fontsize=11
)
ax3.set_xlabel(f"Token chunk start index  (chunk size = {CHUNK})")
ax3.set_ylabel("Mean vertex displacement")
ax3.set_xticks(x)
ax3.set_xticklabels([f"{i*CHUNK}" for i in range(n_chunks)], fontsize=8)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

profile_path = os.path.join(args.out, "ablation_profile.png")
plt.savefig(profile_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {profile_path}")

# ── Plot 3: 3D scatter — baseline mesh colored by "which chunk owns it" ──
# For each vertex, find which ablated mesh moves it most → assign chunk
print("\nGenerating vertex ownership map ...")
ownership = np.full(len(baseline_verts), -1, dtype=int)   # -1 = unassigned
max_displacement_per_vert = np.zeros(len(baseline_verts))

for ci, res in enumerate(chunk_results):
    if res.get('failed'):
        continue
    start = res['start']
    end   = res['end']
    
    z_abl = z.clone()
    z_abl[:, start:end, :] = 0.0
    m = decode_to_mesh(z_abl)
    if m is None or len(m.vertices) != len(baseline_verts):
        continue
    
    diff = np.linalg.norm(
        np.array(m.vertices) - baseline_verts, axis=1
    )
    improved = diff > max_displacement_per_vert
    ownership[improved] = ci
    max_displacement_per_vert[improved] = diff[improved]

# Scatter plot — color by owning chunk
fig3 = plt.figure(figsize=(12, 10))
ax4 = fig3.add_subplot(111, projection='3d')

cmap = plt.cm.get_cmap('tab20', n_chunks)
for ci in range(n_chunks):
    mask = ownership == ci
    if mask.sum() == 0:
        continue
    pts = baseline_verts[mask]
    ax4.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        s=1, c=[cmap(ci)] * mask.sum(),
        label=f"tokens {ci*CHUNK}-{(ci+1)*CHUNK}",
        alpha=0.6,
    )

unassigned = ownership == -1
if unassigned.sum() > 0:
    pts = baseline_verts[unassigned]
    ax4.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                s=0.5, c='gray', alpha=0.2, label='unassigned')

ax4.set_title(
    "Vertex Ownership Map\n"
    "Each color = token chunk that most controls that vertex",
    fontsize=11
)
ax4.set_xlabel('X'); ax4.set_ylabel('Y (height)'); ax4.set_zlabel('Z')
ax4.legend(
    loc='upper left', fontsize=6,
    markerscale=4, ncol=2,
    bbox_to_anchor=(0, 1)
)
ax4.view_init(elev=20, azim=45)

ownership_path = os.path.join(args.out, "vertex_ownership_map.png")
plt.savefig(ownership_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {ownership_path}")

# ── Pattern report ────────────────────────────────────────────────────────
report_lines = [
    "=" * 60,
    f"LATENT ABLATION REPORT",
    f"Mesh:         {args.mesh}",
    f"VAE:          {args.vae}/{args.subfolder}",
    f"Latent shape: ({LATENT_TOKENS}, {LATENT_DIM})",
    f"Chunk size:   {CHUNK}  ({n_chunks} chunks total)",
    "=" * 60,
    f"\nPATTERN DETECTED: {pattern}",
    "",
    pattern_desc,
    "",
    "─" * 60,
    "PER-CHUNK DOMINANT REGION:",
    "",
]

for res in chunk_results:
    if res.get('failed'):
        report_lines.append(f"  tokens {res['label']}: DECODE FAILED")
        continue
    d = res['displacements']
    report_lines.append(
        f"  tokens {res['start']:3d}-{res['end']:3d}: "
        f"dominant={res['dominant']:15s}  "
        f"top={d.get('top_third',0):.4f}  "
        f"mid={d.get('middle_third',0):.4f}  "
        f"bot={d.get('bottom_third',0):.4f}"
    )

report_lines += [
    "",
    "─" * 60,
    "SUGGESTED PART_TOKEN_INDICES (fill in after reviewing plots):",
    "",
    "PART_TOKEN_INDICES = {",
    "    'internal_cavity': list(range(???, ???)),",
    "    'connector':       list(range(???, ???)),",
    "    'external_bottom': list(range(???, ???)),",
    "    'external_top':    list(range(???, ???)),",
    "}",
    "",
    "─" * 60,
    "OUTPUT FILES:",
    f"  ablation_summary.png      — heatmap of chunk vs region displacement",
    f"  ablation_profile.png      — line plot, flat=global / peaked=spatial",
    f"  vertex_ownership_map.png  — 3D mesh colored by controlling chunk",
    f"  chunk_NNN_NNN.obj         — each ablated mesh for Blender inspection",
    "=" * 60,
]

report_text = "\n".join(report_lines)
report_path = os.path.join(args.out, "pattern_report.txt")
with open(report_path, 'w') as f:
    f.write(report_text)

print("\n" + report_text)
print(f"\nSaved: {report_path}")
print(f"\nDone. All outputs in: {args.out}")
