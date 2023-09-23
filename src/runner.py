import torch
import pickle
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier

# if Xgboost is meta-learner:
# from xgboost import XGBClassifier
from dataset.data import Dataset, mode_indx
from m_features import Meta_Features
from metats.pipeline import MetaLearning

torch.manual_seed(2023)
random_state = np.random.RandomState(2023)


class Runner:
    def __init__(self, mode, ExpId, FH, epochs, lr, weightDecay):
        self.ExpId = ExpId  
        self.mode = mode
        self.epochs = epochs
        self.lr = lr
        self.weightDecay = weightDecay
        self.FH = FH
        self._init_mkdir()  
        self.path_base_models = f"base_forecasters/{self.FH}/_all_npy/"  
        self.path_true_d = "src/dataset/_true_npy/"  
        self._check_gpu()  
        _ = self.load_data()
        self.batchsize = 13  
        self.mfeatur_model = Meta_Features(
            self.device,
            self.x_lstm_att.size(2),
            self.x_tcn.size(2),
            self.x_tcn.size(1),
            epochs,
            lr,
            weightDecay,
            self.batchsize,
        )  
        _ = self.load_clf()  
        _ = self._check_mode()

    def run(self):
        _ = self.get_mfeatures()
        pipeline = MetaLearning(method="averaging", loss="mse")
        pipeline.add_metalearner(self.clf)  
        _ = self.load_predictions()  
        if self.train_clf:
            labels = pipeline.generate_labels(self.x_true, self.predictions)
            assert len(set(labels)) == 8, f"the # of labels are not 8!"
            pipeline.meta_learner.fit(X=self.meta_features, y=labels)
        else:
            # test
            weights = pipeline.predict_generate_weights(self.meta_features)
            self.final_forecast = pipeline.averaging_predictions(
                weights, self.predictions
            )
        self.save_results()

    def save_results(self):
        if self.mode == "train":
            if self.FH == 7:
                self.save_models()
            self.save_clf()
        else:
            self.save_forecasts()
            self.call_rmse()

        self.save_idx()

    def call_rmse(self):
        self.rmse = np.sqrt(
            np.mean(np.square(self.x_true - self.final_forecast), axis=1)
        ).mean()

    def save_forecasts(self):
        np.savetxt(
            self.path + f"FH_{self.FH}" + f"_{self.mode}" + ".csv",
            self.final_forecast,
            delimiter=",",
        )

    def save_idx(self):
        np.save(self.path + f"random_idx_{self.mode}_FH_{self.FH}.npy", self.idx)

    def get_mfeatures(self):
        if self.models_gradient:
            self.mfeatur_model.feed_inputs(self.x_tcn, self.x_lstm_att, gradient=True)
        else:
            # load saved parameters
            _ = self.load_models()
            self.mfeatur_model.feed_inputs(self.x_tcn, self.x_lstm_att, gradient=False)

        meta_features = self.mfeatur_model.runner_mlp.latent
        self.meta_features = (
            meta_features.reshape(meta_features.size(0), -1).detach().cpu().numpy()
        )

    def _check_mode(self):
        # it doesn't need to train the deep auto-encoders for all self.FH's
        self.models_gradient = self.mode == "train" and self.FH == 7
        # only clf is needed to be trained for all self.FH's
        self.train_clf = self.mode == "train"

    def load_data(self, initial=False):
        if initial:
            data = Dataset(self.mode)
            # true
            self.x_true = data.x[:, 48:, 0]
            # main
            self.x_lstm_att = data.x_norm[:, :48, 0:1]
            # factors
            self.x_tcn = data.x_norm[:, :48, 1:5]
            # save:
            np.save(self.path_true_d + f"x_true_{self.mode}.npy", self.x_true)
            np.save(
                self.path_true_d + f"x_lstm_att_{self.mode}.npy",
                self.x_lstm_att.numpy(),
            )
            np.save(self.path_true_d + f"x_tcn_{self.mode}.npy", self.x_tcn.numpy())

        else:
            self.x_true = np.load(self.path_true_d + f"x_true_{self.mode}.npy")
            self.x_lstm_att = torch.from_numpy(
                np.load(self.path_true_d + f"x_lstm_att_{self.mode}.npy")
            )
            self.x_tcn = torch.from_numpy(
                np.load(self.path_true_d + f"x_tcn_{self.mode}.npy")
            )

        # adjust x_true based on self.FH
        self.x_true = self.x_true[:, -self.FH :]

        # select random time series
        self.idx = random_state.randint(
            0, self.x_true.shape[0], int(self.x_true.shape[0] / 6)
        )
        self.x_true = self.x_true[self.idx, :]
        self.x_lstm_att = self.x_lstm_att[self.idx, :, :]
        self.x_tcn = self.x_tcn[self.idx, :, :]

    def _check_gpu(self):
        # run on GPU if available:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_predictions(self, initial=False):
        # create (# time series, # models, horizon) predictions
        # for each mode = ["train","valid","test"]
        if initial:
            predictions = self.read_predictions(ifsave=True)
        else:
            predictions = np.load(self.path_base_models + f"base_{self.mode}.npy")
        self.predictions = predictions[self.idx, :, :]

    def read_predictions(self, ifsave=True):
        (start, end) = mode_indx(self.mode)
        models = [
            m for m in os.listdir(f"base_forecasters/{self.FH}") if m != "_all_npy"
        ]
        slot_i = []
        for i in range(start, end):
            base_is = []
            for model in models:
                base_i = np.genfromtxt(
                    f"base_forecasters\{self.FH}\{model}\{model}_{i}.csv", delimiter=","
                )[1:, 1:]
                base_is.append(np.expand_dims(base_i, axis=1))
            is_model = np.concatenate(base_is, axis=1)
            slot_i.append(is_model)
        slots = np.concatenate(slot_i, axis=0)
        if ifsave:
            np.save(self.path_base_models + f"base_{self.mode}.npy", slots)
        return slots

    def save_models(self):
        torch.save(
            self.mfeatur_model.runner_tcn.model.state_dict(), self.path + "model_tcn.pt"
        )
        torch.save(
            self.mfeatur_model.runner_lstm_att.model.state_dict(),
            self.path + "model_lstm_att.pt",
        )
        torch.save(
            self.mfeatur_model.runner_mlp.model.state_dict(), self.path + "model_mlp.pt"
        )

    def load_models(self):
        self.mfeatur_model.runner_tcn.model.load_state_dict(
            torch.load(self.path + "model_tcn.pt")
        )
        self.mfeatur_model.runner_lstm_att.model.load_state_dict(
            torch.load(self.path + "model_lstm_att.pt")
        )
        self.mfeatur_model.runner_mlp.model.load_state_dict(
            torch.load(self.path + "model_mlp.pt")
        )

    def _init_mkdir(self):
        __root_path = os.getcwd()
        path = os.path.join(__root_path, "results/")
        path += "ExpId_" + str(self.ExpId) + "/"
        path += "epochs_" + str(self.epochs) + "_"
        path += "WD_" + str(self.weightDecay) + "_"
        path += "lr_" + str(self.lr)
        path += "/"
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        pass

    def save_clf(self):
        pickle.dump(self.clf, open(self.path + f"clf_FH{self.FH}.pickle", "wb"))

    def load_clf(self):
        if self.mode == "train":
            self.clf = RandomForestClassifier()  # XGBClassifier()
        else:
            self.clf = pickle.load(open(self.path + f"clf_FH{self.FH}.pickle", "rb"))


if __name__ == "__main__":
    (mode, ExpId, FH, epochs, lr, weightDecay) = ("train", "01", 7, 1, 0.01, 0.009)
    arg = (mode, ExpId, FH, epochs, lr, weightDecay)
    runner = Runner(*arg)
    _ = runner.run()