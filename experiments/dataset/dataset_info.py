dataset_size = {
    "MNLI-matched": 392703,
    "SST-2": 67350,
    "QNLI": 104744,
    "QQP": 363847,
    "E2E": 42061,
    "MNIST": 60000,
    "CIFAR10": 60000,
}

classification_steps={
    "MNLI-matched": 1146,
    "SST-2": 192,
    "QNLI": 306,
    "QQP": 1062,
    "E2E": 410,
    "MNIST": 938,
    "CIFAR10": 938,
} # classification tasks - 6 epoches; generation tasks (E2E) - 10 epoches;

# running experiments information
batchsize={
    "MNLI-matched": 2048,
    "SST-2": 2048,
    "QNLI": 2048,
    "QQP": 2048,
    "E2E": 16,
    "MNIST": 64,
    "CIFAR10":2000, 
}