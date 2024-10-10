This repo facilitates my attempts at replicating "Optimal ANN-SNN
conversion for high-accuracy and ultra-low-latency spiking neural
networks".

| SRC    | SEED | ARCH     | DATASET  | L | ANN   | T=1   | T=2   | T=4   | T=8   | T=16  | T=32  | T=64  | T=128 |
|--------|------|----------|----------|---|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| GDrive |      | VGG16    | CIFAR100 | 8 | 77.41 | 58.60 | 64.82 | 70.50 | 74.79 | 76.75 | 76.87 | 77.10 |       |
| Mine   | 42   | VGG16    | CIFAR100 | 8 | 77.14 | 43.07 | 52.74 |       |       |       |       | 77.05 |       |
| Mine   | 42   | ResNet18 | CIFAR100 | 8 | 79.59 | 39.53 | 54.20 | 67.04 | 75.47 | 79.34 | 79.93 | 80.18 |       |
| Mine   | 100  | ResNet18 | CIFAR100 | 8 | 79.73 | 43.64 | 58.15 | 69.13 | 76.66 | 78.93 | 79.78 | 80.04 | 80.09 |
