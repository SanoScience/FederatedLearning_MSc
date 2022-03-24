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


def run_single_experiment(local_epochs, batch_size, clients_count, ff, lr, optimizer, mf, rounds=15):
    output = subprocess.check_output(
        ['sbatch', 'server.sh', str(clients_count), str(rounds), 'FedAvg', str(local_epochs), str(lr),
         str(batch_size), optimizer, str(ff), str(mf)])
    print('sbatch:', output)
    result = re.search('Submitted batch job (\d*)', output.decode('utf-8'))
    print(result.groups())
    server_job_id = result.group(1)
    status = ''
    while status != 'R':
        squeue_output = None
        attempts = 5
        while attempts:
            try:
                ps = subprocess.Popen(('squeue',), stdout=subprocess.PIPE)
                squeue_output = subprocess.check_output(('grep', server_job_id), stdin=ps.stdout)
                ps.wait()
                attempts = 0
            except Exception as e:
                print("Retrying in 30 seconds. Error:", str(e))
                time.sleep(30)
                attempts -= 1
        if not squeue_output:
            raise Exception("Couldn't extract the server's job id")

        split = squeue_output.split()
        status = split[4].decode('utf-8')
        job_id = split[0]
        node = split[7]
        print(f"{job_id}:{status}")
        time.sleep(60)
    print("Starting all clients in 100s!")
    time.sleep(100)
    output = subprocess.check_output(['./run_clients.sh', node.decode('utf-8'), str(clients_count)])
    print(output)
    print("Starting next job in 1 hour.")
    time.sleep(60 * 60)


for optimizer in ['Adagrad', 'Adam']:
    for bs in [8, 4]:
        for le in [1, 2, 3]:
            for ff in [0.5, 0.75, 1.0]:
                run_single_experiment(local_epochs=le, batch_size=bs, clients_count=8, ff=ff, lr=0.001,
                                      optimizer=optimizer, mf=8 * ff)
