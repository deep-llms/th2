import copy
from pathlib import Path
import tempfile
import unittest

from staged_study import MODELS, stage_command, average_states, make_soup


class StagedTests(unittest.TestCase):
    def test_two_stage_parent_recipe_and_discarded_smokes(self):
        for model in MODELS:
            root = Path('/fresh')
            parent = Path('/original/base450/last.pt')
            one = stage_command(model,Path('/cache'),root/'stage256',parent,256,30)
            two = stage_command(model,Path('/cache'),root/'stage512',root/'stage256/last.pt',512,40)
            for cmd,crop,epochs,batch,init in [(one,256,30,32,parent),(two,512,40,16,root/'stage256/last.pt')]:
                get=lambda k:cmd[cmd.index(k)+1]
                for k,v in {'--crop':crop,'--rrc-size':crop,'--epochs':epochs,'--batch-size':batch,
                            '--seed':1,'--data-seed':43,'--split-seed':42,'--views':2,'--warmup-epochs':0}.items():
                    self.assertEqual(get(k),str(v))
                self.assertEqual(get('--init-from'),str(init))
                self.assertEqual(get('--lr'),'0.0001')
                self.assertEqual(get('--sched'),'cosine')
                self.assertEqual('--crop-upscale' in cmd,crop==512)
                self.assertNotIn('smoke',get('--init-from'))

    def test_exact_average_and_rejections(self):
        import torch
        a={'weight':torch.tensor([1.,5.]),'bn.running_var':torch.tensor([1.,3.]),
           'bn.num_batches_tracked':torch.tensor(4)}
        b={'weight':torch.tensor([3.,9.]),'bn.running_var':torch.tensor([3.,5.]),
           'bn.num_batches_tracked':torch.tensor(8)}
        c=average_states(a,b)
        torch.testing.assert_close(c['weight'],torch.tensor([2.,7.]),rtol=0,atol=0)
        torch.testing.assert_close(c['bn.running_var'],torch.tensor([2.,4.]),rtol=0,atol=0)
        self.assertEqual(c['bn.num_batches_tracked'].item(),8)
        self.assertEqual(a['weight'][0].item(),1.)
        bad=copy.deepcopy(b);bad['weight'][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'Nonfinite'):average_states(a,bad)
        bad=copy.deepcopy(b);bad['weight']=bad['weight'].double()
        with self.assertRaisesRegex(ValueError,'layout'):average_states(a,bad)
        with self.assertRaisesRegex(ValueError,'keys'):average_states(a,{'x':torch.ones(2)})

    def test_real_model_soup_reload_and_architecture_guard(self):
        import torch
        from models import build_model,count_trainable
        torch.set_num_threads(1)
        model=build_model(MODELS[0]).eval()
        args={'model':MODELS[0],'norm':'imagenet','split_seed':42,'val_frac':.1,
              'train_split':'train','crop':512,'seed':1,'data_seed':43}
        ck={'model':model.state_dict(),'epoch':40,'args':args,'val_acc':.9}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=root/'direct.pt';b=root/'staged.pt';out=root/'soup.pt'
            torch.save(ck,a);torch.save(ck,b)
            info=make_soup(MODELS[0],a,b,out)
            loaded=torch.load(out,weights_only=False)
            self.assertNotIn('val_acc',loaded)
            self.assertFalse(info['bn_recalibrated'])
            self.assertEqual(loaded['n_params'],count_trainable(model))
            for k,v in model.state_dict().items():torch.testing.assert_close(v,loaded['model'][k],rtol=0,atol=0)
            net=build_model(MODELS[0]).eval();net.load_state_dict(loaded['model'])
            with torch.no_grad():
                x=torch.randn(2,3,32,34)
                torch.testing.assert_close(model(x),net(x),rtol=0,atol=0)
            with self.assertRaisesRegex(ValueError,'architecture'):
                make_soup(MODELS[1],a,b,root/'wrong.pt')


if __name__=='__main__':unittest.main()
