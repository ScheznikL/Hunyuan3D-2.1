import argparse, os, sys, json, subprocess
sys.path.append("/dcs/pg24/u5745134/Desktop/dev/Hunyuan3D-2.1/hy3dshape/tools/render")  # folder containing render.py
from types import SimpleNamespace
from render import main, init_scene  # use functions from your existing render.py

def get_size_gb(path):
    """Return folder size in GB."""
    import subprocess
    out = subprocess.check_output(["du", "-s", "-B1G", path]).split()
    return int(out[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_list", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--uuid_log", type=str, required=True)
    parser.add_argument("--size_limit", type=float, default=650)
    parser.add_argument("--min_models", type=int, default=1000)
    parser.add_argument('--views', type=int, default=24, 
        help='JSON string of views. Contains a list of {yaw, pitch, radius, fov} object.')
    parser.add_argument('--resolution', type=int, default=512, 
        help='Resolution of the images.')
    parser.add_argument('--geo_mode', action='store_true', 
                help='Geometry mode for rendering.')

    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    #args = parser.parse_args()

    with open(args.file_list, "r") as f:
        obj_paths = [line.strip() for line in f if line.strip()]

    success_uuids = []
    for i, obj_path in enumerate(obj_paths, 1):
        uuid = os.path.splitext(os.path.basename(obj_path))[0]
        output_dir = os.path.join(args.output_root, uuid, "render_cond")

        os.makedirs(output_dir, exist_ok=True)

        try:
            # Reset scene to avoid memory leaks
            init_scene()
            #Call render.py's main
            # arg = {
            # 'object': obj_path,
            # 'output_folder': output_dir,
            # 'geo_mode': True,
            # 'resolution': 512,
            # "engine" : "CYCLES",
            # "views" : 12,
            # "save_depth" : True,
            # "save_normal" : True,
            # "save_albedo" : True,
            # "save_mr" : True,
            # "save_mist" : False,
            # "split_normal" : False,
            # "save_mesh" : False
            # }
            arg = SimpleNamespace(
                object=obj_path,
                output_folder=output_dir,
                geo_mode=True,
                resolution=args.resolution,
                engine="CYCLES",
                views=args.views,

                # IMPORTANT: geo_mode-safe flags
                save_depth=False,
                save_normal=False,
                save_albedo=False,
                save_mr=False,
                save_mist=False,
                split_normal=False,
                save_mesh=True
            )
            # class Arg:
            #     'object' = obj_path
            #     output_folder = output_dir
            #     geo_mode = True
            #     resolution = 512
                
            print(f"[INFO] passing to main such args: {arg}")
            main(arg)
            success_uuids.append(uuid)
            print(f"[SUCCESS] {uuid}")
        except Exception as e:
            print(f"[ERROR] {uuid}: {e}", file=sys.stderr)

        # Check size limit after first MIN_MODELS
        if i >= args.min_models:
            if os.path.exists(args.output_root):
                import subprocess
                out = subprocess.check_output(["du", "-sh", args.output_root]).decode()
                if not out:
                    print(f'[WARNING] exiting with out equal {out}')
                    exit(0)
                    #return 0  # Or handle error
                #return int(out[0])

                size_gb = int(out.split()[0].replace("G",""))
                if size_gb >= args.size_limit:
                    print(f"[STOP] Reached {size_gb} GB limit after {i} models")
                    break

    # Save UUID log
    with open(args.uuid_log, "w") as f:
        f.write("\n".join(success_uuids))
