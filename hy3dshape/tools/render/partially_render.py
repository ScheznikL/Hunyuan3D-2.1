import argparse, os, sys, subprocess
from types import SimpleNamespace

RENDER_DIR = "/dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/hy3dshape/tools/render"
sys.path.append(RENDER_DIR)

from render import main, init_scene


def get_size_gb(path):
    out = subprocess.check_output(["du", "-s", "-B1G", path]).split()
    return int(out[0])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_list", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--uuid_log", required=True)
    parser.add_argument("--size_limit", type=int, default=650)
    parser.add_argument("--min_models", type=int, default=1000)

    parser.add_argument("--views", type=int, default=24)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--geo_mode", action="store_true")

    argv = sys.argv[sys.argv.index("--") + 1:]
    return parser.parse_args(argv)


def main_loop(args):
    with open(args.file_list) as f:
        obj_paths = [l.strip() for l in f if l.strip()]

    success = []

    for i, obj_path in enumerate(obj_paths, 1):
        uuid = os.path.splitext(os.path.basename(obj_path))[0]
        out_dir = os.path.join(args.output_root, uuid, "render_cond")

        done_flag = os.path.join(out_dir, ".done")
        lock_flag = os.path.join(out_dir, ".lock")

        if os.path.exists(done_flag) or os.path.exists(lock_flag):
            print(f"[SKIP] {uuid} already has done_flag")
            continue
  # ---- PRE-RENDER COMPLETION CHECK ----
        if os.path.exists(out_dir):
            png_files = [
                f for f in os.listdir(out_dir)
                if f.lower().endswith(".png")
            ]

            if len(png_files) >= args.views:
                # Mark as done if not already
                if not os.path.exists(done_flag):
                    with open(done_flag, "w") as f:
                        f.write("ok\n")
                print(f"[SKIP] {uuid} already has {len(png_files)} PNGs")
                success.append(uuid)
                continue

        os.makedirs(out_dir, exist_ok=True)

        # acquire lock (atomic)
        try:
            with open(lock_flag, "x"):
                pass
        except FileExistsError:
            continue

        render_args = SimpleNamespace(
            object=obj_path,
            output_folder=out_dir,
            engine="CYCLES",
            resolution=args.resolution,
            views=args.views,
            geo_mode=args.geo_mode,

            # geo-safe
            save_depth=False,
            save_normal=False,
            save_albedo=False,
            save_mr=False,
            save_mist=False,
            split_normal=False,
            save_mesh=True,
        )

        try:
            init_scene()
            main(render_args)
            success.append(uuid)
            with open(done_flag, "w") as f:
                f.write("ok\n")
            print(f"[OK] {uuid}")

        except Exception as e:
            print(f"[FAIL] {uuid}: {e}", file=sys.stderr)

        finally:
            if os.path.exists(lock_flag):
                os.remove(lock_flag)

        # disk safety check
        if i >= args.min_models:
            size_gb = get_size_gb(args.output_root)
            if size_gb >= args.size_limit:
                print(f"[STOP] Dataset reached {size_gb} GB")
                break

    return success


if __name__ == "__main__":
    args = parse_args()
    done = main_loop(args)

    with open(args.uuid_log, "w") as f:
        f.write("\n".join(done))
