from main import run
from src.utils import get_config, log_results

if __name__ == '__main__':
    config_path = 'configs/cifar_10_2.yml'
    conf = get_config(config_path)

    max_dataset_size_per_user = 25
    train_batch_size = 10
    conf['data_params']['max_dataset_size_per_user'] = max_dataset_size_per_user
    conf['data_params']['train_batch_size'] = train_batch_size
    min_dataset_size_per_user = [5, 10]
    n_clients_with_min_datasets = [10, 50, 90]

    shard_per_user = [2, 5]

    for shards in shard_per_user:
        conf['data_params']['specific_dataset_params']['classes_per_user'] = shards
        for min_d in min_dataset_size_per_user:
            conf['data_params']['min_dataset_size_per_user'] = min_d
            for n_cl in n_clients_with_min_datasets:
                conf['data_params']['n_clients_with_min_datasets'] = n_cl
                output = run(conf)
                log_results(config_path=config_path, config=conf, output=output)

    config_path = 'configs/cifar_100_5.yml'
    conf = get_config(config_path)

    max_dataset_size_per_user = 25
    train_batch_size = 10
    conf['data_params']['max_dataset_size_per_user'] = max_dataset_size_per_user
    conf['data_params']['train_batch_size'] = train_batch_size
    min_dataset_size_per_user = [5, 10]
    n_clients_with_min_datasets = [10, 50, 90]

    shard_per_user = [5, 20]

    for shards in shard_per_user:
        conf['data_params']['specific_dataset_params']['classes_per_user'] = shards
        for min_d in min_dataset_size_per_user:
            conf['data_params']['min_dataset_size_per_user'] = min_d
            for n_cl in n_clients_with_min_datasets:
                conf['data_params']['n_clients_with_min_datasets'] = n_cl
                output = run(conf)
                log_results(config_path=config_path, config=conf, output=output)
