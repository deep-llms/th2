"""Stage1b FWE data admission, reusing Stage1's document shuffle unchanged."""
import argparse,json
from pathlib import Path
from scripts.a2_stage1_data import prepare
from scripts.a2_prepare_corpora import sha
SPEC='b2121a4f2859f99db421273082c6be035dc5e833af6d1d071f5ed4cd381ed483'
BASE='9ebe0d0a377f0107f7058bf5e4899b5380cbe5f4f518b09091ac533d2c9f254e'
MANIFEST='0685a790f3c60bf0c02f2e0679910ada71d9c3fefd95a99c53578b201b87f3dd'
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 root=Path(__file__).resolve().parents[1];spec=root/'docs/A2_stage1b_fwe_spec.md';base=root/'docs/A2_stage1_spec.md'
 if sha(spec)!=SPEC or sha(base)!=BASE:raise ValueError('Preregistration changed')
 receipt=root/'docs/results/a2_corpora_20261029/fwe_final/verification.json'
 release=json.loads((root/'resources/a2_stage1b_fwe_release_20261030.json').read_text())
 if release['status']!='verified' or not release['all_selected_files_byte_identical_to_original_release']:raise ValueError('Release unverified')
 prepare(root/'resources/a2_corpora_30b_20261029.json',MANIFEST,a.output,base,corpus='fineweb_edu',prior_verification=receipt,replication_spec=spec)
if __name__=='__main__':main()
