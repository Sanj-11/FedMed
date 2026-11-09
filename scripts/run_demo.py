"""Run a self-contained synthetic three-hospital FedAvg demonstration."""
import torch
from model.network import SmallSegmentationModel
from model.train import train_one_step
from federated.parameters import get_model_parameters, set_model_parameters
from federated.fedavg import aggregate_parameters


def main():
    torch.manual_seed(42)
    global_model=SmallSegmentationModel()
    clients=[]
    for idx in range(3):
        model=SmallSegmentationModel(); model.load_state_dict(global_model.state_dict())
        gen=torch.Generator().manual_seed(100+idx)
        x=torch.randn(2,1,32,32,generator=gen); y=torch.randint(0,2,(2,32,32),generator=gen)
        loss=train_one_step(model,x,y)
        clients.append((get_model_parameters(model),len(x)))
        print(f'Hospital {idx+1}: local synthetic training loss={loss:.4f}')
    aggregated=aggregate_parameters([x[0] for x in clients],[x[1] for x in clients])
    set_model_parameters(global_model,aggregated)
    print('FedAvg aggregation completed for 3 simulated hospitals.')
    print('NOTE: synthetic smoke test only; no real MRI dataset or clinical metrics used.')

if __name__=='__main__': main()
