"""
main.py - one entry point for the whole project.

    python main.py check              verify dataset + create train/val/test split
    python main.py train              tune, evaluate and save the PNN
    python main.py predict <image>    classify one image in the terminal
    python main.py app                launch the Streamlit app (FruitFresh AI)
    python main.py all                check -> train
"""

import subprocess
import sys


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1].lower()

    if cmd == "check":
        from src import preprocessing
        manifest = preprocessing.verify_dataset(preprocessing.build_manifest())
        splits = preprocessing.make_splits(manifest)
        print("Saved:", preprocessing.save_splits(splits))
    elif cmd == "train":
        from src import train
        train.main()
    elif cmd == "all":
        main_args = sys.argv
        sys.argv = [main_args[0], "check"]
        main()
        sys.argv = [main_args[0], "train"]
        main()
    elif cmd == "predict":
        if len(sys.argv) != 3:
            print("Usage: python main.py predict <image_path>")
            return
        sys.argv = ["src.predict", sys.argv[2]]
        import runpy
        runpy.run_module("src.predict", run_name="__main__")
    elif cmd == "app":
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    else:
        print(f"Unknown command '{cmd}'.\n{__doc__}")


if __name__ == "__main__":
    main()