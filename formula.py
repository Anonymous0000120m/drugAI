from rdkit import Chem
from rdkit.Chem import Draw

def save_molecule_image(smiles, filename):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        Draw.MolToFile(mol, filename, size=(300, 300))
        print(f'Molecule image saved as {filename}')
    else:
        print('Invalid SMILES representation')

# 用法示例
save_molecule_image('CCO', 'molecule.png')
