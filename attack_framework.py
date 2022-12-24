from tqdm import tqdm

class Attack:
    def __init__(self, substitute_model):
        """

        Parameters
        ----------
        substitute_model : the model architecture that will be trained by the attacker
        """

        self.substitute_model = substitute_model
        
    def run(self, victim_model, n_queries, train_config, dataset=None, n_epoch = 1, batch_size = 4, **kwargs):
        """

        Parameters
        ----------
        victim_model : the model to be stolen
        n_queries : number of queries to be performed on the victim model
        init_ds : (optional) data required by the attack
        kwargs : (optional) additional parameters required by the attack

        Returns
        -------
        The model trained by the attacker
        """
        label_list = []
        for i in tqdm(range(n_queries), desc='number of queries'):
          print(i % n_queries)
          label_list.append(self._run(victim_model, dataset[i % n_queries], **kwargs).detach().argmax())
        

        self.substitute_model = self.train(self.substitute_model.eval(), train_config['num_epochs'], dataset, label_list, train_config['optimizer'], train_config['scheduler'], train_config['loss_fn'], train_config['batch_size'], **kwargs)
        return self.substitute_model

    def _run(self, victim_model, **kwargs):
        raise NotImplementedError
    
    def train(self, substitute_model, num_epochs, dataset, label_list, optimizer, scheduler, loss_fn, n_epoch = 1, batch_size = 4, **kwargs):
      raise NotImplementedError 