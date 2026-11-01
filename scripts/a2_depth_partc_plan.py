"""Assemble C only after actual Part B terminal handoff and the posted A+B report."""
import copy
from efficient_llm.a2_depth.common import SPEC_SHA


def assemble(base,references,history,downstream,predecessor,note,report):
    required={'root','burn_script','supervisor','burn','complete','source_manifest','prior_status','known_burn_identities'}
    if not required<=predecessor.keys() or not predecessor['known_burn_identities']:
        raise ValueError('Actual B terminal/native handoff required')
    if note.get('spec_sha256')!=SPEC_SHA or not note.get('posted_at'):
        raise ValueError('Actual A+B report must be posted before assembling C')
    if history['heal']!=base['heal_streams']['dclm']['path']:
        raise ValueError('DT6 historical stream differs')
    cfg={k:copy.deepcopy(base[k]) for k in ('assets','data_root','data_manifest','small_indices',
         'reference_nll','heal_streams','phase0_sets','layer_sets','deviations')}
    cfg.update(spec_sha256=SPEC_SHA,stage='DT6+PartC+downstream',predecessor=copy.deepcopy(predecessor),
        part_ab_note=copy.deepcopy(note),part_ab_report=copy.deepcopy(report),
        stage1_dclm_reference=copy.deepcopy(references['datasets']['dclm']),stream_history=copy.deepcopy(history),
        prepared=copy.deepcopy(downstream['prepared']),prior_downstream=copy.deepcopy(downstream['prior_downstream']['dclm']),
        downstream_common=copy.deepcopy(downstream['downstream_common']),
        final_sequences=2461,final_targets_per_sequence=4095,resamples=10000,seed=0,
        scientific_policy='log/flag/continue; no time termination',expected_node_hours=39)
    return cfg
