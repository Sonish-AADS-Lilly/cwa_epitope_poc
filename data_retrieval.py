import requests
from typing import Optional, Tuple
import ssl
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def get_uniprot_sequence(uniprot_id: str) -> str:
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    response = requests.get(url, timeout=30, verify=False)
    response.raise_for_status()
    lines = response.text.strip().split('\n')
    return ''.join(lines[1:])

def get_alphafold_structure(uniprot_id: str) -> Optional[str]:
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
    try:
        response = requests.get(url, timeout=30, verify=False)
        if response.status_code == 200:
            return response.text
    except:
        pass
    return None

def get_pdb_structure(uniprot_id: str) -> Optional[str]:
    search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                "operator": "exact_match",
                "value": uniprot_id
            }
        },
        "return_type": "entry"
    }
    
    try:
        response = requests.post(search_url, json=query, timeout=30, verify=False)
        if response.status_code == 200:
            results = response.json()
            if results.get("result_set"):
                pdb_id = results["result_set"][0]["identifier"]
                pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
                pdb_response = requests.get(pdb_url, timeout=30, verify=False)
                if pdb_response.status_code == 200:
                    return pdb_response.text
    except:
        pass
    return None

def get_structure_content(uniprot_id: str) -> Tuple[Optional[str], str]:
    pdb_content = get_pdb_structure(uniprot_id)
    if pdb_content:
        return pdb_content, "solved"
    
    alphafold_content = get_alphafold_structure(uniprot_id)
    if alphafold_content:
        return alphafold_content, "alphafold"
    
    return None, "none"
