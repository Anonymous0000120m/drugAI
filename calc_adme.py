from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

def calculate_adme_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    adme_properties = {}

    # Calculate Lipinski's descriptors
    adme_properties['MW'] = Descriptors.MolWt(mol)
    adme_properties['LogP'] = Descriptors.MolLogP(mol)
    adme_properties['NumHDonors'] = Lipinski.NumHDonors(mol)
    adme_properties['NumHAcceptors'] = Lipinski.NumHAcceptors(mol)

    # Other ADME-related properties can be calculated here

    return adme_properties

# Example SMILES strings of drug molecules
smiles_list = ["CCO", "CCN(CC)C(=O)NC1=C(C=C2C(=C1)C(=NC=N2)N)N", "CC(=O)OC1=CC=CC=C1C(=O)O"]

# Generate ADME dataset
adme_dataset = []
for smiles in smiles_list:
    adme_properties = calculate_adme_properties(smiles)
    if adme_properties is not None:
        adme_dataset.append(adme_properties)

# Print the generated dataset
for i, adme_properties in enumerate(adme_dataset, 1):
    print(f"Molecule {i}:")
    for prop, value in adme_properties.items():
        print(f"{prop}: {value}")
    print()
