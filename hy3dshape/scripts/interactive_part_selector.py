"""
interactive_part_selector.py

Since ShapeVAE is Pattern C (global encoding), we define parts
directly on the mesh geometry, then encode each part separately.

This tool lets you:
  1. View the mesh as a 3D point cloud
  2. Draw Y-axis (height) threshold sliders to define part boundaries
  3. Define up to 4 parts by height bands
  4. Export each part as a submesh + surface tensor
  5. Encode each part separately via ShapeVAE
  6. Save part latents + config for use in PROSTHETIC_PART_CONFIG

TWO MODES:
  --mode slice   : define parts by Y-axis height bands (interactive sliders)
  --mode inspect : just visualize the mesh colored by spatial region

USAGE:
  # Interactive slice mode (requires display / X11 forwarding):
  python interactive_part_selector.py --mesh prior.obj --mode slice

  # Headless mode: define slices via CLI args, export directly:
  python interactive_part_selector.py \
      --mesh prior.obj \
      --mode headless \
      --slices 0.33 0.66 \
      --out ./part_outputs

  # Just inspect the mesh:
  python interactive_part_selector.py --mesh prior.obj --mode inspect
"""

import os
import sys
import json
import argparse
import numpy as np
import trimesh
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button, CheckButtons
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, './hy3dshape')
# ── Args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--mesh',     required=True)
parser.add_argument('--mode',     default='slice',
                    choices=['slice', 'inspect', 'headless'])
parser.add_argument('--out',      default='./part_outputs')
parser.add_argument('--slices',   nargs='+', type=float, default=[0.33, 0.66],
                    help='Y-axis split points as fractions 0-1 (headless mode)')
parser.add_argument('--vae',      default='tencent/Hunyuan3D-2mini')
parser.add_argument('--subfolder',default='hunyuan3d-vae-v2-mini-withencoder')
parser.add_argument('--encode',   action='store_true',
                    help='Encode each part via ShapeVAE after export')
parser.add_argument('--axis',     default='y', choices=['x', 'y', 'z'],
                    help='Axis to slice along (y = height for prosthetic)')
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

if args.mode in ['slice', 'inspect']:
    matplotlib.use('TkAgg')   # interactive — needs display
else:
    matplotlib.use('Agg')     # headless

# ── Part config ───────────────────────────────────────────────────────────
PART_NAMES   = ['external_top', 'external_shell', 'internal_cavity', 'connector']
PART_COLORS  = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']
PART_STRESS  = [0.10,       0.25,      0.95,       0.80]

AXIS_IDX = {'x': 0, 'y': 1, 'z': 2}[args.axis]

# ── Load mesh ─────────────────────────────────────────────────────────────
print(f"Loading mesh: {args.mesh}")
mesh = trimesh.load(args.mesh, force='mesh')
verts = np.array(mesh.vertices)   # [V, 3]
faces = np.array(mesh.faces)      # [F, 3]

axis_vals = verts[:, AXIS_IDX]
a_min, a_max = axis_vals.min(), axis_vals.max()
a_range = a_max - a_min

print(f"  vertices={len(verts)}  faces={len(faces)}")
print(f"  {args.axis}-axis range: [{a_min:.4f}, {a_max:.4f}]  span={a_range:.4f}")

# Downsample for display (keep max 8000 points for speed)
MAX_DISPLAY = 8000
if len(verts) > MAX_DISPLAY:
    idx = np.random.choice(len(verts), MAX_DISPLAY, replace=False)
    display_verts = verts[idx]
else:
    display_verts = verts
    idx = np.arange(len(verts))

# ── Color vertices by part assignment ─────────────────────────────────────
def assign_parts(verts_all, thresholds_frac):
    """
    thresholds_frac: list of fractions [0-1], e.g. [0.33, 0.66]
    Returns: part_ids array [V], int 0..n_parts-1
    """
    thresholds_abs = [a_min + t * a_range for t in sorted(thresholds_frac)]
    part_ids = np.zeros(len(verts_all), dtype=int)
    for i, thresh in enumerate(thresholds_abs):
        part_ids[verts_all[:, AXIS_IDX] > thresh] = i + 1
    return part_ids

def part_colors_array(part_ids):
    hex_colors = PART_COLORS[:len(set(part_ids))]
    colors = np.array([
        matplotlib.colors.to_rgb(hex_colors[min(pid, len(hex_colors)-1)])
        for pid in part_ids
    ])
    return colors

# ── INSPECT MODE ──────────────────────────────────────────────────────────
if args.mode == 'inspect':
    print("Inspect mode — showing mesh colored by Y-thirds...")
    part_ids = assign_parts(display_verts, [0.33, 0.66])
    colors   = part_colors_array(part_ids)

    fig = plt.figure(figsize=(12, 10))
    ax  = fig.add_subplot(111, projection='3d')
    ax.scatter(
        display_verts[:, 0],
        display_verts[:, 1],
        display_verts[:, 2],
        c=colors, s=1, alpha=0.7
    )
    ax.set_title(f"Mesh inspection\n{args.mesh}", fontsize=10)
    ax.set_xlabel('X'); ax.set_ylabel('Y (height)'); ax.set_zlabel('Z')

    from matplotlib.patches import Patch
    legend = [Patch(color=PART_COLORS[i], label=f"part_{i}")
              for i in range(3)]
    ax.legend(handles=legend)
    out_path = os.path.join(args.out, "inspect.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.show()
    sys.exit(0)

# ── HEADLESS MODE ─────────────────────────────────────────────────────────
if args.mode == 'headless':
    final_thresholds = sorted(args.slices)
    print(f"Headless mode — using thresholds: {final_thresholds}")

# ── INTERACTIVE SLICE MODE ────────────────────────────────────────────────
elif args.mode == 'slice':
    print("\nInteractive slice mode:")
    print("  Drag sliders to define part boundaries along the Y axis.")
    print("  Each color band = one mesh part.")
    print("  Click [Export Parts] when satisfied.\n")

    # Initial thresholds
    init_thresholds = [0.25, 0.55, 0.80]   # 3 sliders → 4 parts

    fig = plt.figure(figsize=(15, 11))
    fig.suptitle(
        "Prosthetic Part Selector\n"
        "Drag sliders to set part boundaries  |  Click Export when done",
        fontsize=12
    )

    gs = gridspec.GridSpec(
        2, 2,
        left=0.05, right=0.95,
        top=0.88,  bottom=0.22,
        wspace=0.3, hspace=0.3
    )

    # Main 3D view
    ax3d = fig.add_subplot(gs[:, 0], projection='3d')
    # Side view (Y vs Z projection)
    ax_side = fig.add_subplot(gs[0, 1])
    # Top view (X vs Z projection)
    ax_top  = fig.add_subplot(gs[1, 1])

    def update_views(thresholds):
        part_ids = assign_parts(display_verts, thresholds)
        colors   = part_colors_array(part_ids)

        ax3d.cla()
        ax3d.scatter(
            display_verts[:, 0],
            display_verts[:, 1],
            display_verts[:, 2],
            c=colors, s=1, alpha=0.6,
        )
        # Draw threshold planes
        x_range = [verts[:,0].min(), verts[:,0].max()]
        z_range = [verts[:,2].min(), verts[:,2].max()]
        for ti, t in enumerate(sorted(thresholds)):
            y_val = a_min + t * a_range
            xx, zz = np.meshgrid(x_range, z_range)
            yy = np.full_like(xx, y_val)
            ax3d.plot_surface(xx, yy, zz, alpha=0.15,
                              color=PART_COLORS[ti+1])
            ax3d.text(x_range[0], y_val, z_range[1],
                      f"  thresh {t:.2f}\n  y={y_val:.3f}",
                      fontsize=7, color=PART_COLORS[ti+1])

        ax3d.set_title("3D view", fontsize=9)
        ax3d.set_xlabel('X', fontsize=7)
        ax3d.set_ylabel(f'{args.axis} (slice axis)', fontsize=7)
        ax3d.set_zlabel('Z', fontsize=7)
        ax3d.view_init(elev=20, azim=45)

        # Side view
        ax_side.cla()
        ax_side.scatter(
            display_verts[:, 2],   # Z
            display_verts[:, 1],   # Y (height)
            c=colors, s=0.5, alpha=0.5
        )
        for t in sorted(thresholds):
            y_val = a_min + t * a_range
            ax_side.axhline(y_val, color='black', linewidth=1.2,
                            linestyle='--', alpha=0.8)
        ax_side.set_xlabel('Z'); ax_side.set_ylabel(f'{args.axis}')
        ax_side.set_title('Side view (Z vs Y)', fontsize=9)

        # Top view
        ax_top.cla()
        ax_top.scatter(
            display_verts[:, 0],   # X
            display_verts[:, 2],   # Z
            c=colors, s=0.5, alpha=0.5
        )
        ax_top.set_xlabel('X'); ax_top.set_ylabel('Z')
        ax_top.set_title('Top view (X vs Z)', fontsize=9)

        # Part size summary
        n_parts = len(thresholds) + 1
        counts  = [(part_ids == i).sum() for i in range(n_parts)]
        total   = len(part_ids)
        summary = "  ".join([
            f"Part{i}({PART_NAMES[min(i,len(PART_NAMES)-1)]}): "
            f"{counts[i]/total*100:.1f}%"
            for i in range(n_parts)
        ])
        fig.texts = [t for t in fig.texts
                     if not getattr(t, '_is_summary', False)]
        txt = fig.text(0.5, 0.14, summary, ha='center', fontsize=8,
                       color='#333333')
        txt._is_summary = True

        fig.canvas.draw_idle()

    # ── Sliders ──────────────────────────────────────────────────────────
    slider_axes = [
        plt.axes([0.15, 0.10 - i*0.03, 0.65, 0.02])
        for i in range(len(init_thresholds))
    ]
    sliders = [
        Slider(
            ax=slider_axes[i],
            label=f'Split {i+1}',
            valmin=0.0, valmax=1.0,
            valinit=init_thresholds[i],
            color=PART_COLORS[i+1],
        )
        for i in range(len(init_thresholds))
    ]

    # Track current thresholds
    current_thresholds = list(init_thresholds)

    def on_slider_change(val):
        for i, s in enumerate(sliders):
            current_thresholds[i] = s.val
        update_views(current_thresholds)

    for s in sliders:
        s.on_changed(on_slider_change)

    # ── Export button ─────────────────────────────────────────────────────
    ax_export = plt.axes([0.75, 0.02, 0.15, 0.04])
    btn_export = Button(ax_export, 'Export Parts', color='#2ecc71')

    def on_export(event):
        final_thresholds[:] = sorted(current_thresholds)
        print(f"\n[Export] thresholds = {final_thresholds}")
        plt.close()

    final_thresholds = list(init_thresholds)
    btn_export.on_clicked(on_export)

    update_views(init_thresholds)
    plt.show()
    # After plt.show() returns, final_thresholds is set

# ── Export parts as submeshes ─────────────────────────────────────────────
print(f"\nExporting parts with thresholds: {final_thresholds}")
part_ids_full = assign_parts(verts, final_thresholds)
n_parts       = len(final_thresholds) + 1

part_names_used = PART_NAMES[:n_parts]
part_meshes     = {}
part_configs    = {}

for pid in range(n_parts):
    pname = part_names_used[pid]
    mask  = part_ids_full == pid   # vertex mask

    # Get faces where ALL 3 vertices belong to this part
    face_mask = mask[faces[:, 0]] & mask[faces[:, 1]] & mask[faces[:, 2]]
    sub_faces = faces[face_mask]

    if len(sub_faces) == 0:
        print(f"  [WARN] Part '{pname}': no faces — threshold may be too tight")
        continue

    # Remap vertex indices
    vert_idx   = np.where(mask)[0]
    idx_remap  = {old: new for new, old in enumerate(vert_idx)}
    sub_verts  = verts[vert_idx]
    sub_faces_remapped = np.array([
        [idx_remap[f] for f in face]
        for face in sub_faces
        if all(v in idx_remap for v in face)
    ])

    sub_mesh = trimesh.Trimesh(
        vertices=sub_verts,
        faces=sub_faces_remapped,
        process=False
    )
    part_meshes[pname] = sub_mesh

    out_path = os.path.join(args.out, f"part_{pid}_{pname}.obj")
    sub_mesh.export(out_path)
    print(f"  Part {pid} '{pname}': {len(sub_verts)} verts, "
          f"{len(sub_faces_remapped)} faces → {out_path}")

    # Compute extents for part_measurements [6] = [ext_x,y,z, centroid_x,y,z]
    extents   = sub_verts.max(0) - sub_verts.min(0)   # [3]
    centroid  = sub_verts.mean(0)                       # [3]
    meas_vec  = np.concatenate([extents, centroid])     # [6]

    y_lo = a_min + (final_thresholds[pid-1] if pid > 0 else 0) * a_range
    y_hi = a_min + (final_thresholds[pid]   if pid < len(final_thresholds) else 1) * a_range

    part_configs[pname] = {
        'part_id':       pid,
        'stress':        PART_STRESS[pid],
        'noise_scale':   1.0 - PART_STRESS[pid],   # inverse of stress
        'token_indices': None,                       # N/A for Pattern C
        'y_range':       [float(y_lo), float(y_hi)],
        'extents':       extents.tolist(),
        'centroid':      centroid.tolist(),
        'meas_vec_6':    meas_vec.tolist(),
        'n_verts':       int(len(sub_verts)),
        'n_faces':       int(len(sub_faces_remapped)),
        'mesh_file':     f"part_{pid}_{pname}.obj",
        'latent_file':   f"part_{pid}_{pname}_latent.pt",
        'description':   {
            'external_top':    'Upper aesthetic shell — follow image reference',
            'external_shell':  'Lower shell — some style freedom',
            'internal_cavity': 'MUST match stump measurements — almost frozen',
            'connector':       'Load-bearing attachment points',
        }.get(pname, f'Part {pid}'),
    }

# ── Visualize exported parts ──────────────────────────────────────────────
print("\nGenerating part visualization...")
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 7),
                            subplot_kw={'projection': '3d'})
fig2.suptitle("Exported Parts — verify before encoding", fontsize=13)

views = [
    ('3D view',       20, 45),
    ('Front view',     0,  0),
    ('Side view',      0, 90),
]

for ax, (title, elev, azim) in zip(axes2, views):
    for pid, (pname, sub_mesh) in enumerate(part_meshes.items()):
        sv = np.array(sub_mesh.vertices)
        # Downsample for display
        n_disp = min(3000, len(sv))
        disp_idx = np.random.choice(len(sv), n_disp, replace=False)
        ax.scatter(
            sv[disp_idx, 0], sv[disp_idx, 1], sv[disp_idx, 2],
            s=1, c=PART_COLORS[pid],
            alpha=0.6, label=pname
        )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('X', fontsize=7)
    ax.set_ylabel('Y', fontsize=7)
    ax.set_zlabel('Z', fontsize=7)
    ax.view_init(elev=elev, azim=azim)
    if title == '3D view':
        ax.legend(fontsize=7, markerscale=5, loc='upper left')

vis_path = os.path.join(args.out, "parts_visualization.png")
plt.savefig(vis_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {vis_path}")

# ── Optionally encode each part via ShapeVAE ──────────────────────────────
if args.encode:
    print("\nEncoding parts via ShapeVAE...")
    try:
        from hy3dshape.models.autoencoders import ShapeVAE
        from hy3dshape.surface_loaders import SharpEdgeSurfaceLoader

        vae = ShapeVAE.from_pretrained(
            args.vae, subfolder=args.subfolder,
            use_safetensors=False, variant='fp16',
        ).to('cuda').eval()

        loader = SharpEdgeSurfaceLoader(
            num_sharp_points=0,
            num_uniform_points=81920,
        )

        for pname, sub_mesh in part_meshes.items():
            try:
                surface = loader(sub_mesh).to('cuda', dtype=torch.float16)
                with torch.no_grad():
                    z = vae.encode(surface)    # [1, 512, 64]
                latent_path = os.path.join(
                    args.out,
                    part_configs[pname]['latent_file']
                )
                torch.save(z.cpu(), latent_path)
                print(f"  '{pname}': z.shape={z.shape} → {latent_path}")
            except Exception as e:
                print(f"  '{pname}': encode failed — {e}")
    except ImportError as e:
        print(f"  ShapeVAE not available: {e}")
        print("  Run with hy3dshape env active, or skip --encode for now.")
else:
    print("\n(Skipping VAE encode — add --encode flag to encode parts)")

# ── Save config ───────────────────────────────────────────────────────────
config_path = os.path.join(args.out, "part_config.json")
with open(config_path, 'w') as f:
    json.dump(part_configs, f, indent=2)
print(f"\nSaved config: {config_path}")

# ── Print Python config block ready to paste ─────────────────────────────
print("\n" + "="*60)
print("PASTE THIS INTO YOUR prosthetic_parts.py:")
print("="*60)
print("\nPROSTHETIC_PART_CONFIG = {")
for pname, cfg in part_configs.items():
    print(f"    '{pname}': {{")
    print(f"        'stress':      {cfg['stress']},")
    print(f"        'noise_scale': {cfg['noise_scale']},")
    print(f"        'token_indices': None,   # Pattern C — encode separately")
    print(f"        'latent_file': '{cfg['latent_file']}',")
    print(f"        'meas_vec_6':  {cfg['meas_vec_6']},  # [ext_x,y,z, ctr_x,y,z]")
    print(f"        'description': '{cfg['description']}',")
    print(f"    }},")
print("}")
print("\n" + "="*60)
print(f"\nAll outputs in: {args.out}")