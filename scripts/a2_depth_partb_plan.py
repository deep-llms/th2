"""Assemble Part B only from the actual Part A handoff and posted note.

CPU preparation utility; does not inspect/signal GPUs or launch any workload.
"""
import argparse
import copy
from pathlib import Path
from efficient_llm.a2_depth.common import read, write, SPEC_SHA
from efficient_llm.diagnostic_data import sha256


def assemble(base, references, reference_lock, predecessor, note, report):
    required={'root','burn_script','supervisor','burn','complete','source_manifest','prior_status','known_burn_identities'}
    if not required <= predecessor.keys():
        raise ValueError('Actual terminal/native handoff identities are required; no placeholder launch plan')
    if not note.get('posted_at') or note.get('spec_sha256')!=SPEC_SHA:
        raise ValueError('Record the actual Part A note before assembling its successor')
    cfg={key:copy.deepcopy(base[key]) for key in ('assets','data_root','data_manifest','small_indices',
         'reference_nll','heal_streams','phase0_sets','layer_sets','deviations')}
    cfg.update(spec_sha256=SPEC_SHA,stage='DT4+PartB',predecessor=predecessor,
        part_a_note=note,part_a_report=report,stage1_references=reference_lock,
        short_S1_reference={ds:references['datasets'][ds]['0.2B'] for ds in ('dclm','fwe')},
        final_sequences=2461,final_targets_per_sequence=4095,resamples=10000,seed=0,
        scientific_policy='log/flag/continue; no time termination',expected_node_hours=2.7)
    return cfg


def main():
    p=argparse.ArgumentParser()
    for name in ('part-a-config','handoff','note','report-lock','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();reference=Path('resources/a2_depth_stage1_references_20261101.json')
    lock=dict(path='/mnt/local/deep-llms_th2/'+str(reference),sha256=sha256(reference))
    write(a.output,assemble(read(a.part_a_config),read(reference),lock,read(a.handoff),read(a.note),read(a.report_lock)))


if __name__=='__main__':main()
