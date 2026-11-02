 Plan: train the note filters on AWS EC2 with the full MAPS dataset

 Context

 The CNN filter (cnn_filter.py, eval/train_cnn.py), the split schemes (eval/maps.py: enstdk, maps-full) and the baseline trainers are already built, and the unit tests pass. Training on the full MAPS dataset (about 40 GB, 9 pianos) is too slow on the local PC. The torch build there is CPU-only, and the first ByteDance pass over about 27k files would take many hours.

 So we will do all of it in AWS:
 - An EC2 GPU instance with a 150 GB EBS volume downloads MAPS straight from its direct URLs, so nothing passes through the local PC.
 - It backs the extracted data up to S3, runs every training step, and sends back only the small model files.
 - The code comes from GitHub (Jtrevena14/WaveScore).

 Deliverables (repo changes)

 1. AWS_TRAINING.md: the runbook below as a standalone doc in the repo, for repeat runs.
 2. requirements-train.txt: a slimmer, piano-only dependency list. It has piano_transcription_inference, librosa, scikit-learn, mir_eval, joblib and pretty_midi. It leaves out basic-pitch/TensorFlow, which the piano backend doesn't need and which is awkward on newer Pythons, and it leaves out torch, which the Deep Learning AMI already provides with CUDA.
 3. scripts/aws/ (bash, run on the instance):
    - setup_instance.sh: clone or pull the repo, pip install -r requirements-train.txt, download models/piano_note_pedal.pth from PIANO_CHECKPOINT_URL (backends.py), and check that nvidia-smi and torch.cuda.is_available() work.
    - fetch_maps.sh <urls.txt> <s3://bucket/MAPS>: for each URL, run wget -c (with optional --user/--password) into /data/zips, then unzip -n -x '*/ISOL/*' into /data/MAPS. ISOL (single notes) isn't used by the default subsets, and skipping it saves about 25% of the space. It then deletes the zip and runs aws s3 sync /data/MAPS s3://.../MAPS. Downloading one zip at a time keeps peak disk use low, and re-running skips finished work.
    - restore_maps.sh <s3://bucket/MAPS>: aws s3 sync back to /data/MAPS, for a fresh instance or volume.
    - run_training.sh: sets SHEETCREATOR_MAPS=/data/MAPS and runs the full workflow below with --scheme maps-full. Each step writes to logs/<step>.log through tee, and the script stops on the first failure.
 4. eval/train_cnn.py: add --workers N (default: the CPU count) so that missing CQT patches are built in a ProcessPoolExecutor. Without it, librosa CQT over about 27k files runs on one core and becomes the bottleneck. The cache format and behaviour stay the same.

 Runbook (what goes in AWS_TRAINING.md)

 0. One-time AWS setup (console)

 1. Billing: create a Budget with an email alert at about $25, under Billing → Budgets.
 2. GPU quota: go to Service Quotas → EC2 → Running On-Demand G and VT instances and request 8 vCPUs in us-east-1. New accounts start at 0, and approval can take hours to a day.
 3. S3 bucket: create e.g. wavescore-maps-<suffix> in us-east-1. Keep it private (block all public access) and turn off versioning.
 4. IAM role: create wavescore-ec2 for EC2, with S3 read/write on that bucket only. The instance then uses S3 without any access keys stored on it.
 5. Key pair: create wavescore.pem and keep it in C:\Users\<you>\.ssh\. Windows OpenSSH is already installed.

 1. Launch the instance

 - AMI: Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.x (Ubuntu 22.04). It comes with drivers, CUDA, torch and the AWS CLI.
 - Type: g4dn.2xlarge: a T4 GPU, 8 vCPU and 32 GB RAM, about $0.75/h on-demand. The 8 cores make CQT fast, and 32 GB holds all patches in memory.
 - Storage: root gp3 150 GB. That covers the AMI (about 50 GB), one zip at a time, about 30 GB of extracted MAPS without ISOL, and caches of about 10 GB.
 - Security group: SSH (22) from My IP only.
 - IAM instance profile: wavescore-ec2.
 - Connect: ssh -i ~/.ssh/wavescore.pem ubuntu@<public-dns>, then run tmux so jobs survive a dropped connection.