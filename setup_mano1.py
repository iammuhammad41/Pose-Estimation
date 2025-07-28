import shutil
import os
import hashlib
import argparse

def replace_line_in_file(file_path, line_ids, new_lines):
    """ Replace a line in a given file with new lines. """
    line_ids = [i - 1 for i in line_ids]
    with open(file_path, 'r') as old_file:
        lines = old_file.readlines()

    for idx, line_id in enumerate(line_ids):
        lines[line_id] = new_lines[idx] + '\n'

    with open(file_path, 'w') as new_file:
        new_file.writelines(lines)


def md5(file_path):
    """ Returns the MD5 checksum of the file """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def patch_mano_loader():
    # Patch the MANO loader files
    replace_line_in_file('mano/webuser/smpl_handpca_wrapper_HAND_only.py', [23, 26, 66], ['    import pickle', '    from mano.webuser.posemapper import posemap', '    from mano.webuser.verts import verts_core'])
    replace_line_in_file('mano/webuser/lbs.py', [27, 38], ['from mano.webuser.posemapper import posemap', '        from mano.webuser.posemapper import Rodrigues'])


def setup_mano():
    parser = argparse.ArgumentParser(description="Setup the MANO repository files.")
    parser.add_argument("mano_path", type=str, help="Path to MANO repository.")
    args = parser.parse_args()

    files_needed = [
        'models/MANO_RIGHT.pkl',
        'webuser/verts.py',
        'webuser/posemapper.py',
        'webuser/lbs.py',
        'webuser/smpl_handpca_wrapper_HAND_only.py'
    ]

    files_copy_to = [os.path.join('mano', file) for file in files_needed]
    files_needed = [os.path.join(args.mano_path, file) for file in files_needed]

    assert all([os.path.exists(file) for file in files_needed]), "Some files are missing in the provided MANO path"

    for source, destination in zip(files_needed, files_copy_to):
        shutil.copy2(source, destination)

    patch_mano_loader()
    print("MANO setup complete.")

if __name__ == "__main__":
    setup_mano()
