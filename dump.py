import os

def generate_tree(startpath, ignore_dirs, ignore_files, ignore_exts):
    tree_str = "PROJECT STRUCTURE:\n"
    for root, dirs, files in os.walk(startpath):
        # Modifying dirs in-place to skip ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        tree_str += f"{indent}{os.path.basename(root)}/\n"
        subindent = ' ' * 4 * (level + 1)
        
        for f in files:
            # Check against ignore list and extensions
            if f not in ignore_files and not any(f.endswith(ext) for ext in ignore_exts):
                tree_str += f"{subindent}{f}\n"
    return tree_str

def dump_project(output_file="project_dump.txt"):
    script_name = os.path.basename(__file__)
    
    # ADDED '.venv' and 'venv' to ensure virtual environments are ignored
    ignore_dirs = {'.git', 'node_modules', '__pycache__', 'dist', 'build', '.next', '.venv', 'venv'}
    ignore_files = {output_file, script_name}
    ignore_exts = {'.png', '.jpg', '.jpeg', '.gif', '.ico', '.pdf', '.zip', '.exe', '.dll', '.pyc', '.svg'}

    with open(output_file, 'w', encoding='utf-8') as f:
        # 1. Write the Tree Structure
        f.write(generate_tree('.', ignore_dirs, ignore_files, ignore_exts))
        f.write("\n" + "="*80 + "\n")
        f.write("FILE CONTENTS BELOW\n")
        f.write("="*80 + "\n")

        # 2. Write the Contents
        for root, dirs, files in os.walk('.'):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            for file in files:
                if file in ignore_files or any(file.endswith(ext) for ext in ignore_exts):
                    continue
                
                file_path = os.path.join(root, file)
                f.write(f"\n{'='*80}\n")
                f.write(f"FILE: {file_path}\n")
                f.write(f"{'='*80}\n\n")
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as source_file:
                        f.write(source_file.read())
                except Exception as e:
                    f.write(f"ERROR: Could not read {file}. {str(e)}")
                f.write("\n")

if __name__ == "__main__":
    dump_project()
    print("Logic executed. Project mapped (excluding .venv) and dumped to project_dump.txt")