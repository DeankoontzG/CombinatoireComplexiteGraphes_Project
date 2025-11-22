# compile_all_dzn_to_fzn_recursive.py
import subprocess
import sys
from pathlib import Path

def main():
    mzn_file = "crossword_yuck.mzn"
    root_folder = Path("../../instances")  # dossier racine à explorer récursivement

    if not root_folder.is_dir():
        print(f"❌ Dossier introuvable : {root_folder}", file=sys.stderr)
        sys.exit(1)

    # Recherche récursive de tous les fichiers .dzn
    dzn_files = sorted(root_folder.rglob("*.dzn"))
    if not dzn_files:
        print("⚠️ Aucun fichier .dzn trouvé.")
        sys.exit(0)

    print(f"📁 {len(dzn_files)} fichiers .dzn trouvés dans {root_folder} (récursivement)")

    for dzn_file in dzn_files:
        out_file = dzn_file.with_suffix(".fzn")
        cmd = [
            "minizinc",
            "-c",
            "-O0",
            "-o", str(out_file),
            mzn_file,
            str(dzn_file)
        ]

        print(f"\n▶️ Compilation de {dzn_file.relative_to(root_folder)} → {out_file.name}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✅ OK → {out_file}")
        else:
            print(f"❌ Erreur pour {dzn_file} :", file=sys.stderr)
            print(result.stderr, file=sys.stderr)

if __name__ == "__main__":
    main()
