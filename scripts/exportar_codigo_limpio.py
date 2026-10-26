import json
import re
import os

def exportar_codigo_limpio():
    dir_base = os.path.abspath(os.path.dirname(__file__) + '/..')
    nb_path = os.path.join(dir_base, 'simpsons_cnn.ipynb')
    out_path = os.path.join(dir_base, 'simpsons_cnn_limpio.py')

    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    codigo_limpio = []

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            lines = cell['source']
            for line in lines:
                # Remove inline comments or full comment lines
                # Strip full-line comments
                line_str = line.rstrip('\r\n')
                stripped = line_str.strip()
                if stripped.startswith('#'):
                    continue
                # If inline comment, remove if not inside quotes
                # Simple regex for inline comment:
                # keep clean code
                if '#' in line_str:
                    # check if hash is outside quotes
                    in_single = False
                    in_double = False
                    hash_idx = -1
                    for idx, char in enumerate(line_str):
                        if char == "'" and not in_double:
                            in_single = not in_single
                        elif char == '"' and not in_single:
                            in_double = not in_double
                        elif char == '#' and not in_single and not in_double:
                            hash_idx = idx
                            break
                    if hash_idx != -1:
                        line_str = line_str[:hash_idx].rstrip()
                
                # Check magic commands like %matplotlib or !
                if line_str.strip().startswith('%') or line_str.strip().startswith('!'):
                    continue
                
                codigo_limpio.append(line_str)
            codigo_limpio.append('')

    # Clean multiple consecutive blank lines
    texto = '\n'.join(codigo_limpio)
    texto = re.sub(r'\n{3,}', '\n\n', texto).strip() + '\n'

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(texto)

    print(f'Código limpio exportado con éxito en: {out_path} ({len(texto.splitlines())} líneas)')

if __name__ == '__main__':
    exportar_codigo_limpio()
