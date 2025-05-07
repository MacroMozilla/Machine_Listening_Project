from pprint import pprint

from c_train.a_train_with_config_withatt import train_with_config_withatt
from c_train.b_train_with_config_nonatt import train_with_config_nonatt

for datasetcls in ['WavDataset', 'Wav2VecDataset', 'MelSpecDataset'][:]:
    for attbind in ['add']:
        for lossfctncls in ['CosineSimilarityLoss', 'LearnableAdaCosLoss'][:1]:
            for i, latent_dim in enumerate([128]):
                for j, output_dim in enumerate([256]):
                    for dropout in [0.125]:
                        config = {
                            'batch_size': 25,
                            'epochs': 50,
                            'learning_rate': 1e-3,
                            'latent_dim': latent_dim,
                            'output_dim': output_dim,
                            'patience': 15,
                            'dropout': dropout,
                            'datasetcls': datasetcls,
                            'lossfctncls': lossfctncls,
                            'attbind': attbind,
                        }
                        pprint(config)
                        train_with_config_withatt(config)

for datasetcls in ['WavDataset', 'Wav2VecDataset', 'MelSpecDataset'][:]:
    for attbind in ['add']:
        for lossfctncls in ['CosineSimilarityLoss', 'LearnableAdaCosLoss'][:1]:
            for i, latent_dim in enumerate([128]):
                for j, output_dim in enumerate([256]):
                    for dropout in [0.125]:
                        config = {
                            'batch_size': 25,
                            'epochs': 50,
                            'learning_rate': 1e-3,
                            'latent_dim': latent_dim,
                            'output_dim': output_dim,
                            'patience': 15,
                            'dropout': dropout,
                            'datasetcls': datasetcls,
                            'lossfctncls': lossfctncls,
                            'attbind': attbind,
                        }
                        pprint(config)
                        train_with_config_nonatt(config)
