# Training the speaker extraction model on real meetings

The extraction model speakerlens downloads was trained on artificial mixtures. These
scripts fine-tune it on [AliMeeting](https://www.openslr.org/119/)'s training meetings,
where each person also wore a headset: the table microphone is the input, the person's
headset the answer. On AliMeeting it lowered tcpWER from 0.278 to 0.275 (Eval) and from
0.288 to 0.281 (Test), and DER to 0.219 on both; ASCEND was unchanged.

Needs the `all` and `bench` extras (`pip install -e ".[all,bench]"`), a GPU (we used six
24 GB GPUs; about 4 s of audio per example, batch 4 per GPU), about 35 GB of disk for the
prepared data, and the AliMeeting Train archives (about 96 GB; leave them packed):

```bash
# 0. Download the training archives (also listed at https://www.openslr.org/119/)
mkdir -p data/alimeeting && cd data/alimeeting
wget -c https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Train_Ali_near.tar.gz
wget -c https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Train_Ali_far.tar.gz
cd ../..

# 1. Keep only what training needs from the archives (read as streams; safe to rerun)
python training/prepare_ali_train.py data/alimeeting/Train_Ali_near.tar.gz data/ali_train
python training/prepare_ali_train.py data/alimeeting/Train_Ali_far.tar.gz data/ali_train

# 2. Line up headsets with the table microphone and index the training windows
INIT=~/.cache/speakerlens/models/wesep_bsrnn   # the released model (run speakerlens once)
python training/train_tse.py data/ali_train --init $INIT --index-only
python training/train_tse.py data/ali_train --init $INIT --check-alignment   # optional

# 3. Train (20,000 steps; a snapshot every 5,000 in <out>/step_N/)
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun --nproc_per_node 6 training/train_tse.py \
    data/ali_train --init $INIT --out models/tse_ft_run

# 4. Use it
echo "TSE_MODEL=$PWD/models/tse_ft_run/step_20000" >> .env
```

The README's numbers are for `step_20000`, picked on AliMeeting Eval; Test confirmed it.

**License.** AliMeeting is CC-BY-SA-4.0 and the starting model CC-BY-4.0, so a model you
train this way is best shared under CC-BY-SA-4.0, crediting AliMeeting (Yu et al., M2MeT,
ICASSP 2022) and WeSep. Whether trained weights legally count as an adaptation of their
training data is unsettled; sharing alike is the safe reading. Not legal advice.
