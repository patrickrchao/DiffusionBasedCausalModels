
from utils.dropout import dropout_adj, dropout_adj_parents
from models.vaca.vaca import VACA
from utils.optimizers import get_optimizer, get_scheduler


class VACA_PIWAE(VACA):
    """
    VACA trained with PIWAE
    """


    def __init__(self, *args, **kwargs):
        super(VACA_PIWAE, self).__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.estimator_inference = self.estimator.split('_')[0]
        self.estimator_gener = self.estimator.split('_')[1]

        self.estimator = self.estimator.split('_')[0]

    def set_optim_params(self, optim_params, sched_params):
        self.optim_params = optim_params
        self.sched_params = sched_params

    def configure_optimizers(self):
        tmp_params = self.optim_params['params'].copy()
        # tmp_params['lr'] = tmp_params['lr'] * 0.5

        optim_infer = get_optimizer(self.optim_params['name'])(self.model.encoder_params(),
                                                               **self.optim_params['params'])

        optim_gener = get_optimizer(self.optim_params['name'])(self.model.decoder_params(),
                                                               **tmp_params)

        if isinstance(self.sched_params, dict):
            sched_infer = get_scheduler(self.sched_params['name'])(optim_infer, **self.sched_params['params'])
            sched_gener = get_scheduler(self.sched_params['name'])(optim_gener, **self.sched_params['params'])
            sched = [sched_infer, sched_gener]
        else:
            sched = []
        return [optim_infer, optim_gener], sched



    def training_step(self, batch, batch_idx, optimizer_idx):

        if self.dropout_adj > 0.0 and self.current_epoch >= self.dropout_adj_T:
            batch = batch.clone()
            batch.edge_index, batch.edge_attr = dropout_adj(batch.edge_index, batch.edge_attr,
                                                            p=self.dropout_adj, keep_self_loops= self.keep_self_loops)


        if self.dropout_adj_pa > 0.0 and self.current_epoch >= self.dropout_adj_T:
            batch = batch.clone()
            batch.edge_index, batch.edge_attr = dropout_adj_parents(batch.edge_index, batch.edge_attr,
                                                            p=self.dropout_adj_pa, prob_keep_self=self.dropout_adj_pa_prob_keep_self)


        if optimizer_idx == 0:  # Inference
            objective, data = self.model(batch,
                                         estimator=self.estimator_inference,
                                         beta=self.beta*self.get_beta_annealing_factor(self.current_epoch))

            self.log('train_objective_inference', objective.item(), prog_bar=True)
            for key, value in data.items():
                self.log(f'train_{key}_inference', value.item(), prog_bar=True)
        else:  # Generator
            objective, data = self.model(batch, estimator=self.estimator_gener)

            self.log('train_objective_generator', objective.item(), prog_bar=True)
            for key, value in data.items():
                self.log(f'train_{key}_generator', value.item(), prog_bar=True)

        return -objective
