import time, os
from runner import Runner
import wandb


class Optimizer:
    def __init__(self, FH, ExpId):
        self.FH = FH
        self.ExpId = ExpId
    
    def objective(self):      
        
        wandb.init()
        epochs, weightDecay, lr = wandb.config.epochs, wandb.config.weightDecay, round(wandb.config.learning_rate,ndigits=4)
        modes_loss_rmse = {}

        # run model
        # self.runner = Runner(mode,ExpId,FH,epochs,lr,weightDecay)
        for mode in ['train', 'valid', 'test']:
            self.runner = Runner(mode,self.ExpId,self.FH,epochs,lr,weightDecay)
            print(f"\n### start {mode} phase for wd_{weightDecay}_lr_{lr}_epochs_{epochs}:\n")
            self.runner.run()
            
            if mode!="train":
                # save loss for each mode
                modes_loss_rmse[mode] = self.runner.rmse           
            if mode=="valid":
                mlp_mse = self.runner.mfeatur_model.runner_mlp.loss
                tcn_mse = self.runner.mfeatur_model.runner_tcn.loss
                lstm_att_mse = self.runner.mfeatur_model.runner_lstm_att.loss
        # validation and test loss:
        valid_loss_rmse, test_loss_rmse = modes_loss_rmse['valid'], modes_loss_rmse['test']
        # Log model performance metrics to W&B
        wandb.log({"valid_loss_rmse": valid_loss_rmse, "test_loss_rmse": test_loss_rmse, "mlp_mse":mlp_mse, "tcn_mse":tcn_mse,"lstm_att_mse":lstm_att_mse})

os.environ['WANDB_SILENT']="true"
os.environ['WANDB_MODE']="offline"

print("\nstart:\n")
s_time = time.time()

FH = 7
ExpId = '01'

optimizer = Optimizer(FH, ExpId)
sweep_configs = {
    "method": "grid",
    "metric": {"name": "valid_loss_rmse", "goal": "minimize"},
    "parameters": {
        "epochs": {"values": [1]},
        "weightDecay": {"values": [0.001, 0.005, 0.009]},
        # "learning_rate": {"distribution": "uniform", "min": 0.0100, "max": 0.0400},
        "learning_rate":{"values": [0.01, 0.02, 0.03, 0.05, 0.06]},
    },
}

sweep_id = wandb.sweep(sweep_configs, project=f"FH{FH}_{ExpId}")
wandb.agent(sweep_id=sweep_id, function=optimizer.objective, count=5)

print(f"\n total time: {time.time()-s_time :.2f} seconds.")
print("\n finish.")