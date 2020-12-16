# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/bin/bash

db=$1
question=$2

#python predict.py --table_path data/spider/tables.json --db_id ${db} --question ${question} --output serve/preprocessed.json

time python -u predict.py --dataset ./data \
--glove_embed_path ./data/glove.42B.300d.txt \
--epoch 50 \
--loss_epoch_threshold 50 \
--sketch_loss_coefficie 1.0 \
--beam_size 5 \
--seed 90 \
--save serve/eval.json \
--embed_size 300 \
--sentence_features \
--column_pointer \
--hidden_size 300 \
--lr_scheduler \
--lr_scheduler_gammar 0.5 \
--att_vec_size 300 \
--load_model ./saved_model/IRNet_pretrained.model \
--table_path data/spider/tables.json \
--db_id climbing \
--question "how many mountains are there in each country?"

#python sem2SQL.py --data_path ./data --input_path serve/prediction.json --output_path serve/sql.json
