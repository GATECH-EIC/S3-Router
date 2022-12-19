#!/bin/bash

cd ~/fairseq || exit

task=$1

gpu_id=$2

port=$3

exp_id=1
prune_ratio=(0.9 0.92 0.94 0.96)

for pr in ${prune_ratio[*]}
  do
     CUDA_VISIBLE_DEVICES=$gpu_id fairseq-hydra-train task.data=/home/yf22/fairseq/${task}-bin checkpoint.restore_file=/home/yf22/pretrain_model_asr/roberta.base/model.pt hydra.run.dir=outputs/nlp_${task,,}${exp_id} distributed_training.distributed_init_method=tcp://localhost:${port} model._name=roberta_st model.checkpoint_activations=true model.prune_rate=${pr} model.fix_attn=true model.init_score=weight_rank --config-dir examples/roberta/config/finetuning --config-name ${task,,}; 
     exp_id=$(($exp_id+1))
 done


# entitlement_list=("bigbasin_atn_prod" "default_ftw_gpu" "default_vll_gpu" "gpu_prod")
# entitlement_list=("bigbasin_atn_prod" "bigbasin_atn_prod" "bigbasin_atn_prod" "default_ftw_gpu" "default_ftw_gpu" "default_ftw_gpu" "default_vll_gpu" "default_vll_gpu")

# global_id=0

# exp_id=(334_fs_nfa 318_nfa 339 354_360d 355_360d 358_360d)

# for id in ${exp_id[*]}
# # for id in {334..335}
#   do
#      echo "Submitting the job ft${id} to cloud..."

#      ent=${entitlement_list[$global_id%${#entitlement_list[@]}]}

#      echo "Use entitlement $ent"

#      bento console --file on_device_ai/Tools/experimental/depth_shrink/launch_fblearner.py -- --\
#         --main main \
#         --cfg on_device_ai/Tools/experimental/depth_shrink/configs/0122/ft${id}.yaml \
#         --exp-name "ft${id}" \
#         --batch-size 96 \
#         --amp-opt-level O0 \
#         --accumulation-steps 1 \
#         --num-nodes 4 \
#         --n-gpu-per-node 8 \
#         --entitlement $ent

#      global_id=$(($global_id+1))
#  done


# for id in {359..361}
#   do
#      echo "Submitting the job ft${id} to cloud..."

#      ent=${entitlement_list[$global_id%${#entitlement_list[@]}]}

#      echo "Use entitlement $ent"

#      bento console --file on_device_ai/Tools/experimental/depth_shrink/launch_fblearner.py -- --\
#         --main main \
#         --cfg on_device_ai/Tools/experimental/depth_shrink/configs/0125/ft${id}.yaml \
#         --exp-name "ft${id}" \
#         --batch-size 96 \
#         --amp-opt-level O0 \
#         --accumulation-steps 1 \
#         --num-nodes 4 \
#         --n-gpu-per-node 8 \
#         --entitlement $ent

#      global_id=$(($global_id+1))
#  done


# for id in {359..361}
#   do
#      echo "Submitting the job ft${id}_180d to cloud..."

#      ent=${entitlement_list[$global_id%${#entitlement_list[@]}]}

#      echo "Use entitlement $ent"

#      bento console --file on_device_ai/Tools/experimental/depth_shrink/launch_fblearner.py -- --\
#         --main main \
#         --cfg on_device_ai/Tools/experimental/depth_shrink/configs/0125/ft${id}_180d.yaml \
#         --exp-name "ft${id}_180d" \
#         --batch-size 96 \
#         --amp-opt-level O0 \
#         --accumulation-steps 1 \
#         --num-nodes 4 \
#         --n-gpu-per-node 8 \
#         --entitlement $ent

#      global_id=$(($global_id+1))
#  done

# for id in {359..361}
#   do
#      echo "Submitting the job ft${id}_360d to cloud..."

#      ent=${entitlement_list[$global_id%${#entitlement_list[@]}]}

#      echo "Use entitlement $ent"

#      bento console --file on_device_ai/Tools/experimental/depth_shrink/launch_fblearner.py -- --\
#         --main main \
#         --cfg on_device_ai/Tools/experimental/depth_shrink/configs/0125/ft${id}_360d.yaml \
#         --exp-name "ft${id}_360d" \
#         --batch-size 96 \
#         --amp-opt-level O0 \
#         --accumulation-steps 1 \
#         --num-nodes 4 \
#         --n-gpu-per-node 8 \
#         --entitlement $ent

#      global_id=$(($global_id+1))
#  done


# exp_id=(344_360d)

# for id in ${exp_id[*]}
# # for id in {334..335}
#   do
#      echo "Submitting the job ft${id} to cloud..."

#      ent=${entitlement_list[$global_id%${#entitlement_list[@]}]}

#      echo "Use entitlement $ent"

#      bento console --file on_device_ai/Tools/experimental/depth_shrink/launch_fblearner.py -- --\
#         --main main \
#         --cfg on_device_ai/Tools/experimental/depth_shrink/configs/0122_2/ft${id}.yaml \
#         --exp-name "ft${id}" \
#         --batch-size 64 \
#         --amp-opt-level O0 \
#         --accumulation-steps 1 \
#         --num-nodes 4 \
#         --n-gpu-per-node 8 \
#         --entitlement $ent

#      global_id=$(($global_id+1))
#  done

