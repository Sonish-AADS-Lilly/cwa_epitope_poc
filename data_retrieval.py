import requests
import urllib3
from typing import Tuple, Optional

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def get_uniprot_sequence(uniprot_id: str) -> str:
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url, verify=False)
    response.raise_for_status()
    
    lines = response.text.strip().split('\n')
    sequence = ''.join(lines[1:])
    return sequence

def get_structure_content(uniprot_id: str, structure_type: str = "alphafold") -> Optional[str]:
    """Get structure content based on specified type"""
    
    if structure_type == "pdb":
        # Try to get PDB structure first
        pdb_url = f"https://www.uniprot.org/uniprot/{uniprot_id}.txt"
        response = requests.get(pdb_url, verify=False)
        response.raise_for_status()
        
        pdb_id = None
        for line in response.text.split('\n'):
            if line.startswith('DR   PDB;'):
                pdb_id = line.split(';')[1].strip()
                break
        
        if pdb_id:
            try:
                structure_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
                structure_response = requests.get(structure_url, verify=False)
                structure_response.raise_for_status()
                return structure_response.text
            except requests.RequestException:
                pass
    
    # Try AlphaFold structure (latest version v6)
    try:
        alphafold_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v6.pdb"
        alphafold_response = requests.get(alphafold_url, verify=False)
        alphafold_response.raise_for_status()
        return alphafold_response.text
    except requests.RequestException:
        return None
