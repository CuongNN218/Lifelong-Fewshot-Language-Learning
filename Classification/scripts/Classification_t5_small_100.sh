learnrate=(5e-1)
alllambda=(0.25)
allkd=(0.01)
basemodel="t5_small"
numsample=10
numepoch=150
setting="long_order_${numepoch}e_${numsample}_shot"


# add if for model name 
if [ "$basemodel" == "t5_small" ]; then
  modelname="google/t5-v1_1-small"  
else 
  modelname="google/t5-v1_1-large"
fi

echo "Training with ${modelname}"

basecache="/home/nguyen/projects/baselines/Lifelong-Fewshot-Language-Learning/Classification/cache"


for onerate in ${learnrate[@]}
do
  for onelambda in ${alllambda[@]}
  do
    for onekd in ${allkd[@]}
    do  
        echo "------------------------------"
        expname="${basemodel}_${setting}"
        python -m torch.distributed.launch --nproc_per_node 1 --master_port 29702 Classification.py \
          --cuda 1 \
          --lr $onerate \
          --lm_lambda $onelambda \
          --kd_lamda $onekd \
          --weight_decay 1e-5 \
          --max_grad_norm 1.0 \
          --batch_size_per_gpu 2 \
          --valid_size_per_gpu 12 \
          --test_size_per_gpu 16 \
          --gradient_accumulation_steps 4 \
          --max_epoch 300 \
          --num_workers 0 \
          --save_step 200 \
          --eval_step 200 \
          --tosavepath t5_classify_ckpt \
          --exp_name $expname \
          --seed 0 \
          --model T5SentenceClassify \
          --model_name $modelname \
          --train_sample \
          --max_length 128 \
          --adam_epsilon 1e-8 \
          --warmup_steps 0.01 \
          --use_lm_adapted 1 \
          --num_sample_per_class $numsample \
          --lm_adapted_path  /home/nguyen/projects/baselines/Lifelong-Fewshot-Language-Learning/lm_adapted_t5model/torch_ckpt/small/pytorch_model.bin \
          --cache_path $basecache \
          --prompt_number 300 \
          --ifckpt_onlymodel 1

        # echo "++++++++++++++++++++++++++++++"
        # ps aux | grep Classification.py | awk '{print $2}' | xargs kill -9
    done
  done
done



