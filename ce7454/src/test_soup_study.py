import copy
from pathlib import Path
import tempfile
import unittest

from soup_study import MODELS, EXPECTED, validate_args, make_soup, digest
from staged_study import average_states


class SoupTests(unittest.TestCase):
    def test_fixed_pairs_and_recipe_rejection(self):
        self.assertEqual(MODELS, ('grip_nodown', 'srresnet', 'srresnet_stats'))
        self.assertEqual(set(EXPECTED), set(MODELS))
        for hashes in EXPECTED.values():
            self.assertEqual(len(set(hashes)), 2)
            self.assertTrue(all(len(x)==64 for x in hashes))
        ck={'epoch':40,'args':dict(model=MODELS[0], train_split='train',val_frac=.1,split_seed=42,
            crop=512,rrc_size=512,norm='imagenet',epochs=40,init_from='/base450',seed=1,
            data_seed=43,lr=3e-5,optim='adam',sched='cosine',views=2,kd_teacher=None)}
        validate_args(ck,MODELS[0],Path('/base450'),1,3e-5)
        for key,value in [('model','srresnet'),('init_from','/unrelated'),('norm','none'),('crop',256)]:
            bad=copy.deepcopy(ck);bad['args'][key]=value
            with self.assertRaisesRegex(ValueError,'mismatch'):
                validate_args(bad,MODELS[0],Path('/base450'),1,3e-5)

    def test_actual_models_average_reload_without_inherited_metrics(self):
        import torch
        from models import build_model, count_trainable
        torch.set_num_threads(1)
        for model in MODELS:
            with self.subTest(model=model), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);net=build_model(model).eval();state=net.state_dict()
                common=dict(model=model,train_split='train',val_frac=.1,split_seed=42,crop=512,
                    rrc_size=512,norm='imagenet',epochs=40,init_from='/base450',optim='adam',
                    sched='cosine',views=2,kd_teacher=None)
                left=root/'left.pt';right=root/'right.pt';out=root/'soup.pt'
                for i,path in enumerate((left,right)):
                    args={**common,'seed':i,'data_seed':42+i,'lr':1e-4 if i==0 else 3e-5}
                    sd={k:v.clone() for k,v in state.items()}
                    sd['fc.1.bias'].add_(2*i)
                    torch.save({'model':sd,'epoch':40,'args':args,'val_acc':.9,
                                'n_params_training':count_trainable(net)},path)
                record={'parent':{'path':'/base450'},'sources':[str(left),str(right)],
                        'sha256':[digest(left),digest(right)],'weights':[.5,.5]}
                info=make_soup(model,record,out)
                ck=torch.load(out,weights_only=False)
                self.assertNotIn('val_acc',ck);self.assertNotIn('optimizer',ck)
                self.assertFalse(info['bn_recalibrated'])
                expected={k:v.clone() for k,v in state.items()};expected['fc.1.bias'].add_(1)
                for k in expected:torch.testing.assert_close(ck['model'][k],expected[k],rtol=0,atol=0)
                net.load_state_dict(expected,strict=True)
                reloaded=build_model(model).eval();reloaded.load_state_dict(ck['model'],strict=True)
                with torch.no_grad():
                    x=torch.randn(2,3,32,34)
                    torch.testing.assert_close(net(x),reloaded(x),rtol=0,atol=0)
                with self.assertRaises(FileExistsError):make_soup(model,record,out)
                bad=copy.deepcopy(record);bad['parent']['path']='/wrong'
                with self.assertRaisesRegex(ValueError,'mismatch'):make_soup(model,bad,root/'bad.pt')
                bad=copy.deepcopy(record);bad['sha256'][0]='0'*64
                with self.assertRaisesRegex(ValueError,'changed'):make_soup(model,bad,root/'bad.pt')


if __name__=='__main__':unittest.main()
