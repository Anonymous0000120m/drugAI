import csv
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

def calculate_adme_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    adme_properties = {}

    # Calculate Lipinski's descriptors
    adme_properties['SMILES'] = smiles
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

# Output data to CSV file
csv_file = 'adme_properties.csv'
fieldnames = ['SMILES', 'MW', 'LogP', 'NumHDonors', 'NumHAcceptors']
with open(csv_file, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(adme_dataset)

print(f"ADME properties saved to {csv_file}")
