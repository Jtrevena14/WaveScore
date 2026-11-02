# Training the note filters on AWS (full MAPS)

Runs the whole MAPS workflow (ByteDance transcription, filter tuning, classifier and CNN training, test evaluation) on an EC2 GPU instance. MAPS is downloaded straight onto the instance and backed up to S3. Only the small model files come back to the PC.

Sized for **g4dn.xlarge** (T4 GPU, 4 vCPU, 16 GB RAM, ~$0.53/h on-demand in us-east-1) with a **130 GiB gp3** root volume.

Scripts are in `scripts/aws/`:

| Script | What it does |
|---|---|
| `setup_instance.sh` | Clones or pulls the repo (branch `dev`), installs `requirements-train.txt`, downloads the piano checkpoint, adds 8 GB of swap, and checks the GPU |
| `fetch_maps.sh <urls.txt> <s3://bucket/MAPS>` | Downloads one zip at a time, extracts it without `ISOL`, deletes the zip, and syncs to S3. Can be re-run |
| `restore_maps.sh <s3://bucket/MAPS>` | Copies the extracted MAPS back from S3 onto a fresh instance |
| `run_training.sh [s3://bucket]` | Runs the 5 workflow steps, writing a log for each to `logs/`, then backs up the caches, models and logs to S3 |

## How the pipeline works

### The idea: propose, then filter

Two neural networks are involved, and only one of them is trained.

- **ByteDance piano transcription** (`models/piano_note_pedal.pth`, pre-trained on MAESTRO) is **frozen**. It only runs forward: audio in, candidate notes out. Its weights are never changed.
- **The CNN filter** (`cnn_filter.py`) is a separate, small network trained from scratch. It looks at each candidate and scores how likely the note is to have really been played. **Only its weights are trained.**

The CNN can only **remove** notes, so ByteDance's recall is the upper limit. A note ByteDance misses can't be recovered. That's why ByteDance's output goes to the CNN unfiltered. For the piano backend there is no fragment merging (`merge_gap = 0`) and no amplitude threshold (`pipeline.py::default_settings`). ByteDance still applies its own built-in detection threshold, `onset_threshold = 0.3`, set in `piano_transcription_inference/inference.py`.

```
                    training only: MAPS .txt ground truth
                                   │  labels each candidate
                                   ▼  real (1) / phantom (0)
audio ──► ByteDance (frozen) ──► candidate notes ──► CNN filter (trained) ──► keep / drop ──► final notes
```

After training, the ground truth isn't needed. On your own recordings, the CNN decides by itself.

### What MAPS provides

Each recording is a pair with the same file stem:
- `MAPS_<SUBSET>-..._<PIANO>.wav`: the audio.
- `MAPS_<SUBSET>-..._<PIANO>.txt`: the ground truth, one row per played note as `OnsetTime OffsetTime MidiPitch`.

The `.mid` in the zip is not used, because the `.txt` holds the same notes. `ISOL` (isolated single notes) is not extracted. Training uses `UCHO` (chords), `RAND` (random note clusters) and `MUS` (real pieces).

The `maps-full` split (`SCHEMES` in `eval/maps.py`) divides the pianos into three groups:

| Split | Pianos | Used for |
|---|---|---|
| train | the 7 software pianos | the CNN learns from these |
| val | `StbgTGd2` | early stopping and choosing the keep threshold |
| test | `ENSTDkAm`, `ENSTDkCl` (real Yamaha Disklavier) | the final, unseen score |

Both Disklaviers are kept out of training. Am and Cl are the **same physical piano** recorded with an ambient mic and a close mic. Training on one and testing on the other would make the test score look better than it really is. To change the split, add an entry to `SCHEMES`.

### Why ByteDance has to run first

The `.txt` already labels every *real* note. But the CNN has to learn to tell real candidates from **phantoms**: octave and fifth overtones, leakage from a neighbouring key, and short blips. Those phantoms exist **only in ByteDance's output**. They are the negative examples, and the `.txt` can't provide them. Training on ByteDance's candidates also teaches the CNN ByteDance's own quirks, such as slightly early onsets, which are exactly what it will see at inference.

### Stage by stage

| # | Stage | What happens | Code | Output |
|---|---|---|---|---|
| 1 | Get the data | Download each zip, extract it without ISOL, delete the zip, back up to S3 | `scripts/aws/fetch_maps.sh` | `/data/MAPS/**.wav` + `.txt`; a copy in S3 |
| 2 | Candidate notes | Every wav goes through ByteDance on the GPU, **once**. Each candidate has `start`, `stop`, `midi`, `amplitude` (velocity / 127) and `onset_strength` (the peak onset activation near the start) | `eval/cache.py` → `backends.py::_piano` | `data/cache/piano/<stem>.json`, one per file. Every later step reads these instead of re-running ByteDance |
| 3 | Labelling | Each candidate is matched one-to-one against the `.txt`. Same pitch and onset within **50 ms** → **1 (real)**, otherwise **0 (phantom)**. It runs in memory and takes seconds; nobody labels anything by hand | `eval/metrics.py::correct_labels` (`mir_eval` note matching) | 0/1 labels, held in memory |
| 4 | Note features | About a dozen numbers per candidate: loudness, onset strength, duration, pitch, share of the loudest simultaneous note, rank in its chord, ratio to a possible harmonic parent 12/19/24/28 semitones below, louder neighbour a semitone away, and so on | `features.py::note_features` | Feature matrix, in memory |
| 5 | Baseline filters | For comparison: random search over hand-filter rules, and a logistic-regression / gradient-boosting classifier on the features | `eval/tune_filters.py`, `eval/train_classifier.py` | `models/filter_settings_piano.json`, `models/note_filter_piano.joblib` |
| 6 | Spectrogram patches | A constant-Q spectrogram of each file (22.05 kHz, hop 256 ≈ 11.6 ms, 3 bins per semitone). For each candidate, a **131 × 36** patch from 13 semitones below to 29 above the note, and 8 frames before to 24 after its onset. The patch is placed relative to the note's pitch, so "an octave above" is always in the same rows. Built in parallel by `--workers` processes | `cnn_filter.py::compute_cqt`, `cut_patches`; `eval/train_cnn.py::build_split` | `/data/MAPS_cache/patches/piano/<stem>.npz`, plus one in-memory patch array per split (its size in GB is printed) |
| 7 | CNN training | Patch → 3 conv blocks → combined with the standardised features → small MLP → one logit. Loss is `BCEWithLogitsLoss` with `pos_weight` to balance real against phantom, AdamW at lr 1e-3, batch 256, random crop and contrast jitter. After every epoch it scores the val piano, tries keep thresholds from 0.05 to 0.95, and keeps the best epoch. It stops after 8 epochs without improvement (40 at most), then scores the test pianos once | `eval/train_cnn.py` | `models/note_filter_piano.pt`, containing the weights, keep threshold, feature mean/std, patch config, and val/test F1 |
| 8 | Final comparison | Scores the test pianos with each filter: `none`, `default`, `tuned`, `learned`, `cnn` | `eval/evaluate.py` | Precision / recall / F1 table in `logs/5_evaluate.log` |
| 9 | Use it | On the PC, `transcribe.py --model piano`. `--filter auto` picks the CNN when `note_filter_piano.pt` exists. ByteDance proposes, the CNN keeps candidates scoring at or above its threshold, then `chords.py` groups them into chords | `transcribe.py`, `pipeline.py::make_filter`, `cnn_filter.py::apply_cnn_filter` | The final notes and chords |

Stages 2 and 3 happen inside the training scripts. Stage 2 runs during workflow step 1, and stages 3–4 run at the start of each training step, so there's no separate command for them.

### Reading the results

- **Precision**: the share of kept notes that are real. Phantoms lower it.
- **Recall**: the share of real notes that were kept. Missed or wrongly removed notes lower it.
- **F1**: the balance of the two. This is the number every step optimises and reports.

A good CNN filter **raises precision well above `none`** while **losing little recall**. The test row of step 8 is the honest result, because those pianos were never used for training or threshold picking.

### Possible next step: ByteDance sensitivity

Lowering ByteDance's `onset_threshold`, for example from 0.3 to 0.15–0.2, would let it report quieter real notes and raise the recall limit. It also produces many more phantoms, plus more RAM (about 9.4 KB per candidate) and more disk. Measure recall and candidate count at a few thresholds on a `--limit` sample before changing it. Whatever value you train with must also be used in `transcribe.py`, because the CNN learns that threshold's candidates. This is **not implemented yet**.

## 0. One-time AWS setup (console)

1. **Budget**: Billing → Budgets, create one with an email alert at about $25.
2. **GPU quota**: Service Quotas → EC2 → *Running On-Demand G and VT instances*. Set it to **at least 4 vCPUs** in your region; g4dn.xlarge uses 4. New accounts start at 0.
3. **S3 bucket**: for example `wavescore-maps-<suffix>`, in the same region as the instance. Block all public access, and leave versioning off.
4. **IAM role** `wavescore-ec2`, trusted entity EC2, with an inline policy limited to the bucket:
   ```json
   {"Version": "2012-10-17", "Statement": [
     {"Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": "arn:aws:s3:::wavescore-maps-<suffix>"},
     {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"], "Resource": "arn:aws:s3:::wavescore-maps-<suffix>/*"}
   ]}
   ```
5. **Key pair**: `mri-training`, with the private key at `C:\Users\<you>\.ssh\mri-training.pem`.

## 1. Launch the instance

- **AMI**: *Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.x (Ubuntu 22.04)*. It includes the drivers, CUDA, torch and the AWS CLI.
- **Type**: g4dn.xlarge, on-demand.
- **Storage**: root gp3, 130 GiB. The instance also has a 125 GB local NVMe at `/opt/dlami/nvme`. It is wiped on stop, so `fetch_maps.sh` uses it only for the zips.
- **Security group**: SSH (22) from *My IP* only.
- **Advanced → IAM instance profile**: `wavescore-ec2`.
- **Key pair**: `mri-training`.

Connect from PowerShell:
```powershell
ssh -i "$HOME\.ssh\mri-training.pem" ubuntu@<public-dns>
```
If ssh says the key's permissions are too open, run this once:
```powershell
icacls "$HOME\.ssh\mri-training.pem" /inheritance:r /grant:r "$($env:USERNAME):R"
```
On the instance, run everything inside `tmux`, so jobs survive a dropped connection. Reattach with `tmux attach`.

## 2. Set up

```bash
tmux
git clone -b dev https://github.com/Jtrevena14/WaveScore.git ~/WaveScore
bash ~/WaveScore/scripts/aws/setup_instance.sh
```
It should print the T4 from `nvidia-smi` and `cuda True`.

## 3. Get MAPS

Put the direct zip URLs, one per line, in `~/urls.txt`. For example, copy the file over with `scp -i ... urls.txt ubuntu@<dns>:~`. Then run:
```bash
cd ~/WaveScore
# MAPS_USER=... MAPS_PASSWORD=...   if the source needs a login
bash scripts/aws/fetch_maps.sh ~/urls.txt s3://wavescore-maps-<suffix>/MAPS
```
On a later fresh instance, skip the download and run `bash scripts/aws/restore_maps.sh s3://wavescore-maps-<suffix>/MAPS` instead.

## 4. Train

Run a smoke test first. It uses 3 files per piano and subset and takes minutes:
```bash
LIMIT=3 bash scripts/aws/run_training.sh
```
Then run the full workflow:
```bash
bash scripts/aws/run_training.sh s3://wavescore-maps-<suffix>
```
The steps, each logged to `logs/<step>.log`, all use `--scheme maps-full`. See [How the pipeline works](#how-the-pipeline-works) for what each one does.

| Step | Command | Pipeline stages |
|---|---|---|
| 1 | `eval.evaluate --split all --filter none default` | 2 (ByteDance candidates for every file; the longest step), plus baseline scores |
| 2 | `eval.tune_filters` | 3–5 |
| 3 | `eval.train_classifier` | 3–5 |
| 4 | `eval.train_cnn --workers $(nproc)` | 3, 4, 6, 7 |
| 5 | `eval.evaluate --filter none default tuned learned cnn --split test` | 8 |

Memory: step 4 keeps all the training and validation patches in RAM. They take about 9.4 KB per note, and the script prints the size of each split. The test split is loaded only after the training patches are freed. Watch `free -h` in a second tmux pane. If swap fills up, re-run with `--subset UCHO MUS` or a `--limit`.

## 5. Copy the results back (PowerShell on the PC)

```powershell
cd "C:\Users\<you>\Desktop\SheetCreator"
$k = "$HOME\.ssh\mri-training.pem"; $h = "ubuntu@<public-dns>"
scp -i $k "${h}:~/WaveScore/models/note_filter_piano.pt" "${h}:~/WaveScore/models/note_filter_piano.joblib" "${h}:~/WaveScore/models/filter_settings_piano.json" models\
scp -i $k -r "${h}:~/WaveScore/logs" .
```
`python transcribe.py testmp3/2-5-1.mp3 --model piano` now picks up the CNN filter through `--filter auto`.

## 6. Stop paying

- **Stop** the instance when you are not using it. You pay only for the EBS volume, about $10/month for 130 GiB. **Terminate** it when you are done.
- The S3 copy of MAPS costs about $0.70/month for ~30 GB. Delete it if you won't train again.
