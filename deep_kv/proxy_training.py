"""Small HF Trainer adapters for proxy targets, step-level centering and metrics."""
import time
import numpy as np
import torch

from .training import DeepKVTrainer, PilotCallback


def summarize_proxy(statistics, model):
    values = np.asarray(statistics,dtype=np.float64)
    totals = values.sum(0)
    lm, targets, auxiliary_count, *rest = totals.tolist()
    if targets <= 0 or totals[-1] <= 0 or not np.isfinite(values).all():
        raise ValueError('Invalid proxy evaluation statistics')
    result = dict(lm_loss=lm/targets,loss=lm/targets,input_tokens=int(totals[-1]),
                  target_tokens=int(targets),rows=len(values))
    if auxiliary_count:
        cosines = np.asarray(rest[:-1])/auxiliary_count
        result['aux_loss'] = float((1-cosines).mean())
        width = 3 if model.family == 'P3' else 1
        for index, layer in enumerate(model.layers):
            row = cosines[index*width:(index+1)*width]
            result[f'proxy_layer_{layer}_aux_loss'] = float((1-row).mean())
            for c, cosine in enumerate(row):
                result[f'proxy_layer_{layer}_cosine_{c}'] = float(cosine)
    return result


class ProxyCallback(PilotCallback):
    def on_step_begin(self,args,state,control,**kwargs):
        self.started = time.perf_counter()

    def on_step_end(self,args,state,control,**kwargs):
        trainer = self.trainer
        model = trainer.model
        if trainer.center_totals is not None:
            sums, counts = trainer.center_totals
            sums = trainer.accelerator.reduce(sums,reduction='sum')
            counts = trainer.accelerator.reduce(counts,reduction='sum')
            model.update_mu(sums,counts)
            trainer.center_totals = None
        if args.device.type == 'cuda':
            torch.cuda.synchronize(args.device)
        self.seconds_per_update = time.perf_counter()-self.started
        return super().on_step_end(args,state,control,**kwargs)

    def enrich_log(self,logs,state):
        if 'loss' in logs:
            logs['step_lm_loss'] = self.latest['lm_loss']
            logs.update({f'step_{k}':v for k,v in self.latest.items() if k.startswith('proxy_') or k=='aux_loss'})
            logs['seconds_per_update'] = self.seconds_per_update
            logs['proxy_lambda'] = self.trainer.model.auxiliary_weight(max(0,state.global_step-1))
            for name, head in self.trainer.model.heads.items():
                for c, value in enumerate(head.alpha.detach().abs().mean(-1)):
                    logs[f'proxy_layer_{name}_mean_abs_alpha_{c}'] = float(value)


class ProxyTrainer(DeepKVTrainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.compute_metrics = lambda prediction: summarize_proxy(prediction.predictions,self.model)
        # All proxy-screen arms, including A/V, use global token denominators.
        self.model_accepts_loss_kwargs = True
        self.center_totals = None

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
        counts = torch.zeros(2,dtype=torch.long,device=device)
        for inputs in batch_samples:
            ctx = self.context(inputs)
            counts += torch.stack((ctx.targets().sum(),ctx.valid.sum())).to(device)
        return self.accelerator.reduce(counts,reduction='sum')

    def training_step(self,model,inputs,num_items_in_batch=None):
        if not bool(self.model.mu_initialized):
            # Same first microbatch, no loader advancement, no RNG-dependent modules.
            prepared = self._prepare_inputs(inputs)
            with torch.no_grad(), self.accelerator.autocast(), self.compute_loss_context_manager():
                outputs = model(self.context(prepared),compute_auxiliary_losses=False,
                                auxiliary_grad=False,collect_target_statistics=True)
            sums = self.accelerator.reduce(outputs['center_sums'],reduction='sum')
            counts = self.accelerator.reduce(outputs['center_counts'],reduction='sum')
            self.model.update_mu(sums,counts,initialize=True)
        return super().training_step(model,inputs,num_items_in_batch)

    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        training = model.training and self.is_in_train
        weight = self.model.auxiliary_weight(self.state.global_step)
        interval = max(1,self.state.logging_steps or int(self.args.logging_steps))
        diagnostic = (self.state.global_step+1)%interval==0 or (self.args.logging_first_step and self.state.global_step==0)
        compute_aux = bool(self.model.family and (not training or weight>0 or diagnostic))
        outputs = model(self.context(inputs),compute_auxiliary_losses=compute_aux,
                        auxiliary_grad=model.training and weight>0,
                        collect_target_statistics=training and self.model.settings.target_centering)
        counts = (num_items_in_batch if num_items_in_batch is not None else
                  torch.stack((outputs['lm_count'],outputs['aux_count'])))
        loss = outputs['lm_sum']/counts[0].clamp_min(1)
        if weight>0:
            loss = loss+weight*outputs['aux_sum']/counts[1].clamp_min(1)
        if num_items_in_batch is not None:
            loss = loss*self.accelerator.num_processes
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite proxy objective')
        if training:
            values = outputs['statistics'].sum(0)
            self.step_totals = values if self.step_totals is None else self.step_totals+values
            if self.model.family and self.model.settings.target_centering:
                sums,counts = outputs['center_sums'],outputs['center_counts']
                if self.center_totals is None:
                    self.center_totals = (sums.clone(),counts.clone())
                else:
                    self.center_totals[0].add_(sums);self.center_totals[1].add_(counts)
        if self._active_sdpa_audit is not None and loss.requires_grad:
            loss.register_hook(self._active_sdpa_audit.backward_marker)
        return (loss,outputs) if return_outputs else loss

    def evaluate(self,eval_dataset=None,ignore_keys=None,metric_key_prefix='eval'):
        result = super().evaluate(eval_dataset,ignore_keys,metric_key_prefix)
        if self.model.family:
            with self.model.without_proxy():
                ablated = super().evaluate(eval_dataset,ignore_keys,metric_key_prefix+'_no_proxy')
            result.update(ablated)
            key = metric_key_prefix+'_reliance_loss_increase'
            result[key] = ablated[metric_key_prefix+'_no_proxy_lm_loss']-result[metric_key_prefix+'_lm_loss']
            self.log({key:result[key]})
        return result
