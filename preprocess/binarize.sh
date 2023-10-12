# this script is used pre-process fine-tune dialogue corpus
# in this paper we consider three corpus, i.e., dailydialog, dstc7avsd, personachat
# running this script will cause all files with the same name in the `BINARY_DIR` folder will be overwritten

PROJECT_PATH=/Users/lemuria_chen/PycharmProjects/DialogVED
#PROJECT_PATH=/remote-home/wchen/project/DialogVED

USER_DIR=${PROJECT_PATH}/src
VOCAB_PATH=${PROJECT_PATH}/vocab.txt
NUM_WORKERS=20

# pre-process (dailydialog)
########################################################################################################################
DATA_DIR=${PROJECT_PATH}/data/finetune/dailydialog
PROCESSED_DIR=${DATA_DIR}/processed
BINARY_DIR=${DATA_DIR}/binary

"$(which fairseq-preprocess)" \
  --fp16 \
  --user-dir ${USER_DIR} \
  --task ved_translate \
  --source-lang src \
  --target-lang tgt \
  --trainpref ${PROCESSED_DIR}/train \
  --validpref ${PROCESSED_DIR}/valid \
  --testpref ${PROCESSED_DIR}/test \
  --destdir ${BINARY_DIR} \
  --srcdict ${VOCAB_PATH} \
  --tgtdict ${VOCAB_PATH} \
  --workers ${NUM_WORKERS}

# pre-process (dstc7avsd)
########################################################################################################################
DATA_DIR=${PROJECT_PATH}/data/finetune/dstc7avsd
PROCESSED_DIR=${DATA_DIR}/processed
BINARY_DIR=${DATA_DIR}/binary

"$(which fairseq-preprocess)" \
  --fp16 \
  --user-dir ${USER_DIR} \
  --task ved_translate \
  --source-lang src \
  --target-lang tgt \
  --trainpref ${PROCESSED_DIR}/train \
  --validpref ${PROCESSED_DIR}/valid \
  --testpref ${PROCESSED_DIR}/test \
  --destdir ${BINARY_DIR} \
  --srcdict ${VOCAB_PATH} \
  --tgtdict ${VOCAB_PATH} \
  --workers ${NUM_WORKERS}

# pre-process (personachat)
########################################################################################################################
DATA_DIR=${PROJECT_PATH}/data/finetune/personachat
PROCESSED_DIR=${DATA_DIR}/processed
BINARY_DIR=${DATA_DIR}/binary

"$(which fairseq-preprocess)" \
  --fp16 \
  --user-dir ${USER_DIR} \
  --task ved_translate \
  --source-lang src \
  --target-lang tgt \
  --trainpref ${PROCESSED_DIR}/train \
  --validpref ${PROCESSED_DIR}/valid \
  --testpref ${PROCESSED_DIR}/test \
  --destdir ${BINARY_DIR} \
  --srcdict ${VOCAB_PATH} \
  --tgtdict ${VOCAB_PATH} \
  --workers ${NUM_WORKERS}
