"""Fixed P7 tuning contracts; old arm names and checkpoints keep their meaning."""

TARGETGRAD_VARIANTS = {
    'P7-simple-targetgrad-short': dict(lookahead=2),
    'P7-simple-targetgrad-medium': dict(lookahead=3),
    'P7-simple-targetgrad-weighted': dict(weights=(1.6, 1.2, .8, .4)),
    'P7-simple-targetgrad-huber': dict(loss='smooth_l1'),
    'P7-simple-targetgrad-keyonly': dict(entries='keyonly'),
    'P7-simple-targetgrad-valueonly': dict(entries='valueonly'),
    'P7-simple-targetgrad-nonlinear': dict(transform='residual_silu', transform_width=32),
    'P7-simple-targetgrad-sparse': dict(sparse=True),
    'P7-simple-targetgrad-heads2': dict(groups=1),
    'P7-simple-targetgrad-heads8': dict(groups=4),
}
LM_VARIANTS = {
    'P7-simple-lm-sparse-noaux': dict(sparse=True, noaux=True),
    'P7-simple-lm-keyonly-noaux': dict(entries='keyonly', noaux=True),
    'P7-simple-lm-valueonly-noaux': dict(entries='valueonly', noaux=True),
    'P7-simple-lm-valueonly-gated': dict(entries='valueonly', mix_init=.05),
}
TUNING_SPECS = {**TARGETGRAD_VARIANTS, **LM_VARIANTS}
TUNING_ARMS = tuple(TUNING_SPECS)
NOAUX_ARMS = ('P7-simple-lm-noaux',) + tuple(a for a,s in LM_VARIANTS.items() if s.get('noaux'))
KEYONLY_ARMS = ('P7-simple-lm-keyonly',) + tuple(a for a,s in TUNING_SPECS.items() if s.get('entries')=='keyonly')
VALUEONLY_ARMS = ('P7-simple-lm-valueonly',) + tuple(a for a,s in TUNING_SPECS.items() if s.get('entries')=='valueonly')


def settings_overrides(arm):
    """CLI/recipe values required by a named arm; reject mismatches at construction."""
    s = TUNING_SPECS.get(arm, {})
    return dict(proxy_groups=s.get('groups',2), proxy_lookahead=s.get('lookahead',4),
                proxy_loss_form=s.get('loss','cosine'))


def tuning_metadata(arm):
    s = TUNING_SPECS[arm]
    valueonly = s.get('entries')=='valueonly'
    result = dict(auxiliary_routes=dict(shallow=False,target=arm in TARGETGRAD_VARIANTS),
        lm_to_predictor=arm in LM_VARIANTS,
        memory_entries=dict(version=3, keys='native' if valueonly else 'proxy',
            values='native' if s.get('entries')=='keyonly' else 'proxy',
            normalization='joint_softmax_with_native_entries',
            query_heads=2*s.get('groups',2), kv_groups=s.get('groups',2),
            transform=s.get('transform','linear')),
        auxiliary_objective=dict(version=1,kind='none' if s.get('noaux') else s.get('loss','cosine')))
    if valueonly:
        result['memory_entries']['implementation']='average_values_single_attention'
    if 'mix_init' in s:
        result['memory_entries'].update(implementation='gated_values_single_attention',
            normalization='native_softmax', gate='sigmoid_per_query_head', gate_init=s['mix_init'],
            mass_diagnostic='effective_value_mix_not_separate_attention_probability')
    if 'transform_width' in s:
        result['memory_entries'].update(transform_width=s['transform_width'],
            transform_initialization='zero_residual_output', transform_location='after_rms_before_projection')
    if s.get('loss')=='smooth_l1':result['auxiliary_objective'].update(beta=1.,reduction='mean_hidden_then_global_token_mean')
    if 'weights' in s:result.update(quantity='weighted_mlp_window_sum',layer_weights=list(s['weights']))
    return result
