import re
import subprocess
import time

parameters = {'local_epochs': [1, 2, 3, 4, 5],
              'batch_size': [1, 2],
              'optimizers': ['Adam', 'SGD', 'Adagrad'],
              'clients': [2, 3, 4, 5, 6, 7, 8],
              'ff': [0.5, 0.75, 1.0],
              'lr': [0.001, 0.0001],
              'image_size': [256, 512],
              'backbone': ['EfficientNetB4', 'ResNet18']}


def run_single_experiment(local_epochs, batch_size, clients_count, ff, lr, optimizer, backbone, rounds=15):
    output = subprocess.check_output(
        ['sbatch', 'server.sh', str(clients_count), str(rounds), 'FedAvg', str(local_epochs), str(lr), str(batch_size),
         optimizer, str(ff)])
    print('sbatch:', output)
    result = re.search('Submitted batch job (\d*)', output.decode('utf-8'))
    print(result.groups())
    server_job_id = result.group(1)
    status = ''
    while status != 'R':
        ps = subprocess.Popen(('squeue',), stdout=subprocess.PIPE)
        squeue_output = subprocess.check_output(('grep', server_job_id), stdin=ps.stdout)
        ps.wait()
        split = squeue_output.split()
        status = split[4].decode('utf-8')
        job_id = split[0]
        node = split[7]
        print(f"{job_id}:{status}")
        time.sleep(10)
    print("Starting all clients in 20s!")
    time.sleep(20)
    output = subprocess.check_output(['./run_clients.sh', node.decode('utf-8'), str(clients_count)])
    print(output)


run_single_experiment(2, 1, 4, 0.75, 0.001, 'Adagrad', 'EfficientNetB4', rounds=15)
