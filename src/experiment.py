import time, os
from runner import Runner
import wandb


class Optimizer:
    def __init__(self, ExpId):
        self.ExpId = ExpId
    
    def objective(self):      
        wandb.init()
        epochs, weightDecay, lr = wandb.config.epochs, wandb.config.weightDecay, round(wandb.config.learning_rate,ndigits=4)
        modes_loss_rmse = {}

        # run model
        for FH in [7,4,1]:
            print(f"forecast:{FH}")
            for mode in ['train', 'valid', 'test']:
                self.runner = Runner(mode,self.ExpId,FH,epochs,lr,weightDecay)
                print(f"\n### start {mode} phase for wd_{weightDecay}_lr_{lr}_epochs_{epochs}:\n")
                self.runner.run()
                if mode !='train':
                    # save loss for each mode
                    modes_loss_rmse[mode] = self.runner.rmse           
                if mode=="valid" and FH==7:
                    mlp_mse = self.runner.mfeatur_model.runner_mlp.loss
                    tcn_mse = self.runner.mfeatur_model.runner_tcn.loss
                    lstm_att_mse = self.runner.mfeatur_model.runner_lstm_att.loss
            # validation and test loss:
            valid_loss_rmse, test_loss_rmse = modes_loss_rmse['valid'], modes_loss_rmse['test']
            wandb.log({f"valid_loss_rmse_FH{FH}": valid_loss_rmse, f"test_loss_rmse_FH{FH}": test_loss_rmse})    
        # Log model performance metrics to W&B
        wandb.log({"tcn_mse":tcn_mse,"lstm_att_mse":lstm_att_mse,"mlp_mse":mlp_mse})

os.environ['WANDB_SILENT']="true"
# os.environ['WANDB_MODE']="offline"

print("\nstart:\n")
s_time = time.time()

ExpId = '01'

optimizer = Optimizer(ExpId)
sweep_configs = {
    "method": "grid",
    "metric": {"name": "valid_loss_rmse", "goal": "minimize"},
    "parameters": {
        "epochs": {"values": [5]},
        "weightDecay": {"values": [0.001, 0.005, 0.009]},
        # "learning_rate": {"distribution": "uniform", "min": 0.0100, "max": 0.0400},
        "learning_rate":{"values": [0.01, 0.02, 0.03, 0.05, 0.06]},
    },
}

sweep_id = wandb.sweep(sweep_configs, project=f"Retail_M_{ExpId}_colab")
wandb.agent(sweep_id=sweep_id, function=optimizer.objective, count=45)

print(f"\n total time: {time.time()-s_time :.2f} seconds.")
print("\n finish.")