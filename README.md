# 概要

前処理、学習、評価データに対する推論に関するスクリプトが入っています。

# 必要パッケージ
DockerFileを用意できればよかったのですが、環境的に使えなかったためパッケージを列挙します。

- pytorch
- torchaudio
- tqdm
- librosa
- scipy
- numpy
- matplotlib
- accelerate
- cython
- fastdtw
- omegaconf

# 使用方法

以下のスクリプト実行前にconfigs/preprocess.yamlおよび、configs/train.yamlを適切な値に変更してください。  
確認していただきたいものとして、preprocess.yamlの「src_dir」、「tgt_dir」は実行環境によって異なることが想定できるので確認をお願いします。

「src_dir」、「tgt_dir」が正しければ以下のコマンドで学習まで実行します。

```bash
$ sh run.sh
```

なお、段階ごとに実行したい場合は以下の順で実行します。

## 1. 前処理
```
$ python preprocess.py -c configs/preprocess.yaml

-c or --config(default: configs/preprocess.yaml): configファイルの指定
```

## 2. 学習

```
$ python train.yaml -c configs/train.yaml

-c or --config(default: configs/train.yaml): configファイルの指定
```

## 3. (optional)評価データに対する推論

```
$ python validate.py --model_dir ./pretrained/vc \
                     --hifi_gan ./pretrained/hifi_gan \
                     --data_dir ./DATA \
                     --output_dir ./outputs
```

