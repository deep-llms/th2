"""Small HF Trainer adapters for proxy targets, step-level normalization and metrics."""
import time
import numpy as np
import torch

from .training import DeepKVTrainer, PilotCallback


def reduce_moments(accelerator, values, fused):
    if not fused:return tuple(accelerator.reduce(v,reduction='sum') for v in values)
    sizes=[v.numel() for v in values]
    merged=accelerator.reduce(torch.cat([v.reshape(-1) for v in values]),reduction='sum')
    return tuple(v.reshape(original.shape) for v,original in zip(merged.split(sizes),values))


def summarize_proxy(statistics, model):
    values = np.asarray(statistics,dtype=np.float64)
    totals = values.sum(0)
    lm, targets, auxiliary_count, *rest = totals.tolist()
    if targets <= 0 or totals[-1] <= 0 or not np.isfinite(values).all():
        raise ValueError('Invalid proxy evaluation statistics')
    result = dict(lm_loss=lm/targets,loss=lm/targets,input_tokens=int(totals[-1]),
                  target_tokens=int(targets),rows=len(values))
    if auxiliary_count:
        size = len(model.layers)*(3 if model.family == 'P3' else 1)
        cosines = np.asarray(rest[:size])/auxiliary_count
        losses = np.asarray(rest[size:2*size])/auxiliary_count
        relational = np.zeros(size)
        if model.relational_proxy:
            query_count = rest[-2]
            if query_count:
                relational = np.asarray(rest[2*size:3*size])/query_count
            result['relational_queries'] = int(query_count)
            result['cos_loss'] = float(losses.mean())
            result['rel_loss'] = float(relational.mean())
            result['aux_loss'] = float((losses+.5*relational).mean())
        else:
            result['aux_loss'] = float(losses.mean())
        if model.memory_proxy:result['cos_loss'] = float(losses.mean())
        width = 3 if model.family == 'P3' else 1
        for index, layer in enumerate(model.layers):
            row = cosines[index*width:(index+1)*width]
            result[f'proxy_layer_{layer}_aux_loss'] = float(losses[index*width:(index+1)*width].mean())
            if model.memory_proxy:
                result[f'proxy_layer_{layer}_cos_loss'] = float(losses[index])
            if model.relational_proxy:
                result[f'proxy_layer_{layer}_rel_loss'] = float(relational[index])
                result[f'proxy_layer_{layer}_aux_loss'] += .5*float(relational[index])
            for c, cosine in enumerate(row):
                result[f'proxy_layer_{layer}_cosine_{c}'] = float(cosine)
    return result


class ProxyCallback(PilotCallback):
    def on_train_begin(self,args,state,control,**kwargs):
        if self.trainer.model.anticipatory and state.global_step==0:
            self.trainer.log({f'proxy_layer_{name}_mean_abs_alpha_{c}':float(value)
                for name,head in self.trainer.model.heads.items()
                for c,value in enumerate(head.alpha.detach().abs().mean(-1))})
        return control

    def on_step_begin(self,args,state,control,**kwargs):
        self.started = time.perf_counter()

    def on_step_end(self,args,state,control,**kwargs):
        trainer = self.trainer
        model = trainer.model
        if trainer.center_totals is not None:
            sums, squares, counts, clipped = reduce_moments(trainer.accelerator,trainer.center_totals,
                                                          model.anticipatory or model.memory_proxy)
            trainer.normalization_metrics = model.update_statistics(sums,squares,counts)
            trainer.normalization_metrics['clip_fraction'] = clipped/(counts*model.mu.shape[-1])
            trainer.center_totals = None
        if args.device.type == 'cuda':
            torch.cuda.synchronize(args.device)
        self.seconds_per_update = time.perf_counter()-self.started
        if model.memory_proxy and state.global_step==1000:
            control.should_evaluate = True  # Spec's fixed mid-run cosine/mass check.
        return super().on_step_end(args,state,control,**kwargs)

    def enrich_log(self,logs,state):
        if 'loss' in logs:
            logs['step_lm_loss'] = self.latest['lm_loss']
            logs.update({f'step_{k}':v for k,v in self.latest.items() if k.startswith('proxy_') or k=='aux_loss'})
            logs['seconds_per_update'] = self.seconds_per_update
            logs['proxy_lambda'] = self.trainer.model.auxiliary_weight(max(0,state.global_step-1))
            for metric, values in self.trainer.normalization_metrics.items():
                for key, value in zip(self.trainer.model.mean_layers,values.detach().cpu().tolist()):
                    suffix = str(key) if isinstance(key,int) else f'{key[0]}_deep_{key[1]}'
                    logs[f'target_layer_{suffix}_{metric}'] = value
            for name, head in self.trainer.model.heads.items():
                if not hasattr(head,'alpha'):continue
                for c, value in enumerate(head.alpha.detach().abs().mean(-1)):
                    logs[f'proxy_layer_{name}_mean_abs_alpha_{c}'] = float(value)


class ProxyTrainer(DeepKVTrainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.compute_metrics = lambda prediction: summarize_proxy(prediction.predictions,self.model)
        # All proxy-screen arms, including A/V, use global token denominators.
        self.model_accepts_loss_kwargs = True
        self.center_totals = None
        self.normalization_metrics = {}
        self._memory_eval_offset = 0

    def log(self,logs,start_time=None):
        # Enrich before HF copies log_history and dispatches W&B callbacks.
        logs = dict(logs)
        for callback in self.callback_handler.callbacks:
            if isinstance(callback,ProxyCallback):
                callback.enrich_log(logs,self.state)
        return super().log(logs,start_time)

    def summarize_statistics(self,values):
        return summarize_proxy(values,self.model)

    def get_decay_parameter_names(self,model):
        return [name for name in super().get_decay_parameter_names(model) if not name.endswith('.alpha')]

    def _get_num_items_in_batch(self,batch_samples,device):
        if not batch_samples:
            return None
        memory = self.model.relational_proxy
        counts = torch.zeros(3 if memory else 2,dtype=torch.long,device=device)
        offset = 0
        for inputs in batch_samples:
            ctx = self.context(inputs)
            counts[:2] += torch.stack((ctx.targets().sum(),ctx.valid.sum())).to(device)
            if memory:
                from .proxy_memory import eligible_queries
                counts[2] += eligible_queries(ctx.segments).sum(-1).clamp_max(256).sum().to(device)
                # Logical row ordinal within an optimizer update, independent of
                # microbatch boundaries, disjoint across ranks; no mutable RNG.
                inputs['_p7_sequence_indices'] = [
                    (offset+i)*self.accelerator.num_processes+self.accelerator.process_index
                    for i in range(len(ctx.input_ids))]
                offset += len(ctx.input_ids)
        return self.accelerator.reduce(counts,reduction='sum')

    def training_step(self,model,inputs,num_items_in_batch=None):
        if not bool(self.model.mu_initialized):
            # Reuse exactly the same prepared microbatch and weights for both passes.
            prepared = self._prepare_inputs(inputs)
            for phase in ('mean','variance'):
                with torch.no_grad(), self.accelerator.autocast(), self.compute_loss_context_manager():
                    outputs = model(self.context(prepared),compute_auxiliary_losses=False,
                                    auxiliary_grad=False,collect_target_statistics=True,statistics_mode=phase)
                sums, squares, counts = reduce_moments(self.accelerator,
                    tuple(outputs[key] for key in ('center_sums','center_squares','center_counts')),
                    self.model.anticipatory or self.model.memory_proxy)
                self.model.update_statistics(sums,squares,counts,initialize=phase)
        return super().training_step(model,inputs,num_items_in_batch)

    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        training = model.training and self.is_in_train
        weight = self.model.auxiliary_weight(self.state.global_step)
        interval = max(1,self.state.logging_steps or int(self.args.logging_steps))
        diagnostic = (self.state.global_step+1)%interval==0 or (self.args.logging_first_step and self.state.global_step==0)
        # At lambda=0 an isolated estimator still needs zero-gradient DDP hooks.
        isolated_training = training and (self.model.anticipatory or self.model.memory_proxy) and self.model.settings.isolate_estimator
        compute_aux = bool(self.model.family and (not training or weight>0 or diagnostic or isolated_training))
        memory_kwargs = {}
        if self.model.relational_proxy:
            indices = inputs.get('_p7_sequence_indices')
            if indices is None and not training:
                indices = [(self._memory_eval_offset+i)*self.accelerator.num_processes+self.accelerator.process_index
                           for i in range(len(inputs['input_ids']))]
                self._memory_eval_offset += len(indices)
            memory_kwargs = dict(relational_step=self.state.global_step,sequence_indices=indices)
        outputs = model(self.context(inputs),compute_auxiliary_losses=compute_aux,
                        auxiliary_grad=model.training and (weight>0 or isolated_training),
                        collect_target_statistics=training and bool(self.model.family),**memory_kwargs)
        counts = (num_items_in_batch if num_items_in_batch is not None else
                  torch.stack((outputs['lm_count'],outputs['aux_count'])))
        loss = outputs['lm_sum']/counts[0].clamp_min(1)
        if weight>0 or isolated_training:
            loss = loss+weight*outputs['aux_sum']/counts[1].clamp_min(1)
            if self.model.relational_proxy:
                query_count = num_items_in_batch[2] if num_items_in_batch is not None else outputs['rel_count']
                loss = loss+weight*.5*outputs['rel_sum']/query_count.clamp_min(1)
        if num_items_in_batch is not None:
            loss = loss*self.accelerator.num_processes
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite proxy objective')
        if training:
            values = outputs['statistics'].sum(0)
            self.step_totals = values if self.step_totals is None else self.step_totals+values
            if self.model.family:
                moments = tuple(outputs[key] for key in ('center_sums','center_squares','center_counts','clip_counts'))
                if self.center_totals is None:
                    self.center_totals = tuple(value.clone() for value in moments)
                else:
                    for total,value in zip(self.center_totals,moments):total.add_(value)
        if self._active_sdpa_audit is not None and loss.requires_grad:
            loss.register_hook(self._active_sdpa_audit.backward_marker)
        return (loss,outputs) if return_outputs else loss

    def evaluate(self,eval_dataset=None,ignore_keys=None,metric_key_prefix='eval'):
        self._memory_eval_offset = 0
        result = super().evaluate(eval_dataset,ignore_keys,metric_key_prefix)
        if self.model.family:
            with self.model.without_proxy():
                self._memory_eval_offset = 0
                ablated = super().evaluate(eval_dataset,ignore_keys,metric_key_prefix+'_no_proxy')
            result.update(ablated)
            key = metric_key_prefix+'_reliance_loss_increase'
            result[key] = ablated[metric_key_prefix+'_no_proxy_lm_loss']-result[metric_key_prefix+'_lm_loss']
            self.log({key:result[key]})
        if self.model.memory_proxy:
            dataset = self.eval_dataset if eval_dataset is None else eval_dataset
            indices = range(self.accelerator.process_index,min(8,len(dataset)),self.accelerator.num_processes)
            moments = self.model.mu.new_zeros((len(self.model.layers),2))
            # At most eight fixed examples globally. Unwrapped model: ranks with
            # zero examples must not enter a DDP forward collective.
            with torch.no_grad(), self.accelerator.autocast(), self.compute_loss_context_manager():
                for index in indices:
                    inputs = self._prepare_inputs(self.data_collator([dataset[index]]))
                    moments += self.model.proxy_attention_mass(self.context(inputs))
            moments = self.accelerator.reduce(moments,reduction='sum')
            if not bool(torch.isfinite(moments).all()) or bool((moments[:,1]<=0).any()):
                raise ValueError('Invalid P7 attention mass diagnostics')
            mass = {f'{metric_key_prefix}_proxy_layer_{layer}_attention_mass':float(row[0]/row[1])
                    for layer,row in zip(self.model.layers,moments)}
            if self.state.global_step==1000:
                cosine = float(np.mean([result[f'{metric_key_prefix}_proxy_layer_{layer}_cosine_0']
                                        for layer in self.model.layers]))
                mass[f'{metric_key_prefix}_p7_mean_cosine'] = cosine
                mass[f'{metric_key_prefix}_p7_low_cosine_trigger'] = cosine < .3
            result.update(mass)
            self.log(mass)
        return result
