"""Publish the web app to Hugging Face Spaces and print the live link.

1. Create a free account at https://huggingface.co and a token with WRITE access
   (Settings -> Access Tokens -> New token -> type 'Write').
2. Run:   python deploy_hf.py --space YOUR_USERNAME/face-mask-detection --token hf_xxxxxxxx
   (or set the HF_TOKEN environment variable and omit --token)

Your live link will be  https://huggingface.co/spaces/YOUR_USERNAME/face-mask-detection
(the first build takes a few minutes). For camera and screen sharing use the direct app address
https://YOUR_USERNAME-face-mask-detection.hf.space  (it is printed at the end).
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", required=True, help="USERNAME/space-name, e.g. arun/face-mask-detection")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Hugging Face write token")
    ap.add_argument("--private", action="store_true", help="Create the Space as private")
    args = ap.parse_args()
    if not args.token:
        sys.exit("[error] Provide --token hf_xxx (or set HF_TOKEN). See the instructions at the top of this file.")
    if "/" not in args.space:
        sys.exit("[error] --space must look like USERNAME/space-name")

    folder = HERE / "hf_space"
    # always ship the newest app, front end and weights
    (folder / "app.py").write_bytes((HERE / "app.py").read_bytes())
    (folder / "web").mkdir(exist_ok=True)
    (folder / "web" / "index.html").write_bytes((HERE / "web" / "index.html").read_bytes())
    best = HERE / "weights" / "best.pt"
    if best.exists():
        (folder / "best.pt").write_bytes(best.read_bytes())
    if not (folder / "best.pt").exists():
        sys.exit("[error] best.pt missing (weights/best.pt).")

    from huggingface_hub import HfApi

    api = HfApi(token=args.token)
    api.create_repo(args.space, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=args.space, repo_type="space",
                      commit_message="Deploy face mask detection app")
    print(f"\n[done] Uploaded. Live link (ready after the build finishes, a few minutes):\n"
          f"  https://huggingface.co/spaces/{args.space}\n"
          f"  https://{args.space.replace('/', '-').replace('_', '-').lower()}.hf.space")
    return 0


if __name__ == "__main__":
    sys.exit(main())
