import re
import subprocess
import time
import os

parameters = {'local_epochs': [1, 2, 3, 4, 5],
              'batch_size': [1, 2],
              'optimizers': ['Adam', 'SGD', 'Adagrad'],
              'clients': [2, 3, 4, 5, 6, 7, 8],
              'ff': [0.5, 0.75, 1.0],
              'lr': [0.001, 0.0001],
              'image_size': [256, 512],
              'backbone': ['EfficientNetB4', 'ResNet18']}


def run_single_experiment(local_epochs, batch_size, clients_count, ff, lr, optimizer, mf, rounds):
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
    print("Starting all clients in 12m!")
    time.sleep(12 * 60)
    output = subprocess.check_output(['./run_clients.sh', node.decode('utf-8'), str(clients_count)])
    print(output)
    print("Starting next job in 2.5h.")
    time.sleep(60 * 60 * 2.5)


clients_count = 3
for optimizer in ['Adam', 'Adagrad']:
    for lr in [0.001]:
        for bs in [16]:
            for le in [3, 2, 1]:
                for ff in [0.75, 1.0]:
                    rounds = 12
                    mf = int(clients_count * ff)
                    res_dir = f'dp_fpn_resnet18_r_{rounds}-c_{clients_count}_bs_{bs}_le_{le}_fs_FedAvg' \
                              f'_mf_{mf}_ff_{ff}_do_{False}_o_{optimizer}_lr_{lr}_image_{256}_IID'
                    if os.path.exists(res_dir) and os.path.exists(res_dir + "/" + "result.csv"):
                        print("skipping: ", res_dir)
                        continue
                    run_single_experiment(local_epochs=le, batch_size=bs, clients_count=clients_count, ff=ff, lr=lr,
                                          optimizer=optimizer, mf=mf, rounds=rounds)
