import unittest
import torch
from efficient_llm.warm_order.route_gradient import residual,Model
from efficient_llm.warm_order.window_film import Model as Parent
from efficient_llm.warm_order.model import Config


class RouteGradientTests(unittest.TestCase):
    def test_model_parent_identity_causality_and_training_forward(self):
        torch.set_num_threads(1);torch.manual_seed(41)
        c=Config(d=8,heads=2,kv_heads=1,head_dim=4,ff=16,vocab=17,length=9,prefix=2)
        parent=Parent(c,'WL',4,2,0).eval();parent.reference=True
        with torch.no_grad():parent.router.weight.normal_(std=.1);parent.film_gamma.normal_(std=.1)
        control=Model(c,'WL',4,2,0,surrogate=False).eval();new=Model(c,'WL',4,2,0,surrogate=True).eval()
        for model in (control,new):model.load_state_dict(parent.state_dict())
        x=torch.randint(17,(3,9));base=parent.hidden(x)
        for model in (control,new):
            torch.testing.assert_close(base,model.hidden(x),rtol=0,atol=0)
            for cut in (1,2,4,8):
                z=x.clone();z[:,cut:]=(z[:,cut:]+3)%17
                torch.testing.assert_close(base[:,:cut],model.hidden(z)[:,:cut],rtol=0,atol=0)
        parent.train();control.train();new.train();rng=torch.get_rng_state()
        expected=parent.hidden(x)
        for model in (control,new):
            torch.set_rng_state(rng);got=model.hidden(x)
            torch.testing.assert_close(got,expected,rtol=0,atol=0)
            torch.testing.assert_close(model.loads,parent.loads,rtol=0,atol=0)
        control.zero_grad();new.zero_grad();torch.set_rng_state(rng);control.hidden(x).square().mean().backward()
        torch.set_rng_state(rng);new.hidden(x).square().mean().backward()
        self.assertGreater(float((control.router.weight.grad-new.router.weight.grad).abs().max()),1e-9)

    def test_hard_identity_and_independent_analytic_gradient(self):
        torch.manual_seed(29);z=torch.randn(3,4,dtype=torch.float64,requires_grad=True)
        delta=torch.randn(3,4,5,2,dtype=torch.float64,requires_grad=True)
        upstream=torch.randn(3,5,2,dtype=torch.float64);p=z.softmax(-1);choice=p.argmax(-1)
        hard=residual(p,choice,delta,False);surrogate=residual(p,choice,delta,True)
        torch.testing.assert_close(hard,surrogate,rtol=0,atol=0)
        gz,gd=torch.autograd.grad((surrogate*upstream).sum(),(z,delta))
        dot=(delta.detach()*upstream[:,None]).sum((2,3));gprob=p.detach()*dot
        gprob=gprob+torch.nn.functional.one_hot(choice,4)*dot
        expected_z=p.detach()*(gprob-(p.detach()*gprob).sum(1,keepdim=True))
        expected_d=torch.zeros_like(delta)
        for i,k in enumerate(choice):expected_d[i,k]=p.detach()[i,k]*upstream[i]
        torch.testing.assert_close(gz,expected_z,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(gd,expected_d,rtol=0,atol=0)

    def test_alternative_outputs_change_router_gradient_only(self):
        torch.manual_seed(8);choice=torch.tensor([0]);delta=torch.randn(1,4,2,2)
        def run(values):
            z=torch.tensor([[2.,1.,0.,-1.]],requires_grad=True);p=z.softmax(-1)
            out=residual(p,choice,values,True);g,=torch.autograd.grad(out.sum(),z);return out.detach(),g
        out,g=run(delta);changed=delta.clone();changed[:,3]+=7
        second,g2=run(changed)
        torch.testing.assert_close(out,second,rtol=0,atol=0)
        self.assertGreater(float((g-g2).abs().max()),.01)


if __name__=='__main__':unittest.main()
