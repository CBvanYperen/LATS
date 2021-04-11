Please follow the following steps in order to replicate the results presented in this paper:
1. Set up the code to run PEGASUS
2. Download and sort dataset used for our results
3. Use our datasets in the PEGASUS model

Details for each step are outlined below:
# 1. Set up the code to run PEGASUS
Please follow the instructions below which can additionally be found [here](https://github.com/google-research/pegasus/blob/master/README.md).
## 1.1 PEGASUS library

Pre-training with Extracted Gap-sentences for Abstractive SUmmarization
Sequence-to-sequence models, or PEGASUS, uses self-supervised objective Gap
Sentences Generation (GSG) to train a transformer encoder-decoder model. The
paper can be found on [arXiv](https://arxiv.org/abs/1912.08777). ICML 2020 accepted.

If you use this code or these models, please cite the following paper:
```
@misc{zhang2019pegasus,
    title={PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization},
    author={Jingqing Zhang and Yao Zhao and Mohammad Saleh and Peter J. Liu},
    year={2019},
    eprint={1912.08777},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
## 1.2 Setup

### create an instance on google cloud with GPU (optional)

Please create a project first and create an instance

```
gcloud compute instances create \
  ${VM_NAME} \
  --zone=${ZONE} \
  --machine-type=n1-highmem-8 \
  --accelerator type=nvidia-tesla-v100,count=1 \
  --boot-disk-size=500GB \
  --image-project=ml-images \
  --image-family=tf-1-15 \
  --maintenance-policy TERMINATE --restart-on-failure
```

### install library and dependencies

Clone library on github and install requirements.

```
git clone https://github.com/google-research/pegasus
cd pegasus
export PYTHONPATH=.
pip3 install -r requirements.txt
```

Download vocab, pretrained and fine-tuned checkpoints of all experiments from [Google Cloud](https://console.cloud.google.com/storage/browser/pegasus_ckpt).

Alternatively in terminal, follow the instruction and install [gsutil](https://cloud.google.com/storage/docs/gsutil_install). Then

```
mkdir ckpt
gsutil cp -r gs://pegasus_ckpt/ ckpt/

```

## 1.3 Finetuning on downstream datasets

### on existing dataset

Finetune on an existing dataset `aeslc`.

```
python3 pegasus/bin/train.py --params=aeslc_transformer \
--param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model \
--train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 \
--model_dir=ckpt/pegasus_ckpt/aeslc
```

If you would like to finetune on a subset of dataset, please refer to the [example of input pattern](https://github.com/google-research/pegasus/blob/master/pegasus/data/datasets.py#L186).

Evaluate on the finetuned dataset.

```
python3 pegasus/bin/evaluate.py --params=aeslc_transformer \
--param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=5,beam_alpha=0.6 \
--model_dir=ckpt/pegasus_ckpt/aeslc
```

Note that the above example is using a single GPU so the batch_size is much smaller
than the results reported in the paper.

### add new finetuning dataset

Two types of dataset format are supported: [TensorFlow Datasets (TFDS)](https://www.tensorflow.org/datasets) or TFRecords.

[This tutorial](https://www.tensorflow.org/datasets/add_dataset) shows how to add a new dataset in TFDS.
(The fine-tuning dataset is expected to be supervised, please provide
`supervised_keys` in dataset info).

Tfrecords format requires each record to be a tf example of `{"inputs":tf.string, "targets":tf.string}`.

For example, if you registered a TFDS dataset called `new_tfds_dataset` for training and evaluation, and have some files in tfrecord format called `new_dataset_files.tfrecord*` for test, they can be registered in `/pegasus/params/public_params.py`.

```
@registry.register("new_params")
def my_param(param_overrides):
  return public_params.transformer_params(
      {
          "train_pattern": "tfds:new_tfds_dataset,train",
          "dev_pattern": "tfds:new_tfds_dataset,validation",
          "test_pattern": "tfrecord:new_dataset_files.tfrecord*",
          "max_input_len": 512,
          "max_output_len": 128,
          "train_steps": 10000,
          "learning_rate": 0.0001,
          "batch_size": 8,
      }, param_overrides)
```

### Evaluation metrics.

Evaluation results can be found in `mode_dir`. Summarization metrics are automatically
calculated for each evaluation point.

-   [ROUGE](https://www.aclweb.org/anthology/W04-1013.pdf) is the main metric
    for summarization quality.

-   [BLEU](https://www.aclweb.org/anthology/P02-1040.pdf) is an alternative
    quality metric for language generation.

-   [Extractive Fragments Coverage & Density](https://arxiv.org/pdf/1804.11283.pdf)
    are metrics that measures the abstractiveness of the summary.


-   Repetition Rates measures generation repetition failure modes.

-   Length statistics measures the length distribution of decodes comparing to gold summary.

Several types of output files can be found in `model_dir`

-   text_metrics-*.txt: above metrics in text format. Each row contains metric
    name, 95% lower bound value, mean value, 95% upper bound value.
-   inputs-*.txt, targets-*.txt, predictions-*.txt: raw text files of model
    inputs/outputs.


# 2. Download and sort dataset used for our results
User @JafferWilson has provided the processed data, which you can download [here](https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail). (See discussion [here](https://github.com/abisee/cnn-dailymail/issues/9) about why the original authors do not provide it themselves).

1. Download the CNN_STORIES_TOKENIZED and DM_STORIES_TOKENIZED [here](https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail).

2. Extract the downloaded folders and move all ".story" folders into a single folder.

3. Use "Sortdata.py" to create sorted datasets, see further instructions in code comments.

# 3. Use created datasets in the PEGASUS model
Please follow the instructions in step "1.3 Finetuning on downstream datasets", to finetune the provided pre-trained PEGASUS model on the created datasets. Further instructions can be found [here](https://github.com/google-research/pegasus).

An example registry is as follows:
```
@registry.register("cnndm_CLOP_complexity_bucket_1")
def cnn_dailymail(param_overrides):
  return transformer_params(
      {
          "train_pattern": "tfrecord:path/to/dataset/cnndm_CLOP_complexity_bucket_1/cnndm_CLOP_complexity_bucket_1_train.tfrecord",
          "dev_pattern": "tfrecord:path/to/dataset/cnndm_CLOP_complexity_bucket_1/cnndm_CLOP_complexity_bucket_1_validate.tfrecord",
          "test_pattern": "tfds:cnn_dailymail/plain_text-test",
          "max_input_len": 1024,
          "max_output_len": 128,
          "train_steps": 500,
          "learning_rate": 0.001,
          "batch_size": 2,
      }, param_overrides)
```