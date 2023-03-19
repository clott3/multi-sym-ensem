CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python linear_probe_ensem.py --dataset imagenet --data /home/cloh/imagenet_data/ --pretrained /home/cloh/checkpoints/10-16-simclr-rot-0.0-800ep-resnet50.pth /home/cloh/checkpoints/11-26-simclr-rot-0.4-800ep-resnet50.pth /home/cloh/checkpoints/10-18-simclr-rot-inv-800ep-resnet50.pth --weights freeze --exp-id ensem-rot-freeze --multiprocessing-distributed --lr-classifier 1.0


 python eval_ensem_gate.py --gpu 0 --dataset imagenet-100 --data /home/gridsan/groups/MAML-Soljacic/ --pretrained /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth --exp-id=testgate --combine_sep_ckpts

python eval_ensem_gate.py --gpu 0 --server=sc --dataset imagenet --data /home/gridsan/groups/datasets/ImageNet --pretrained /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth --exp-id=testgate --combine_sep_ckpts

python eval_ensem_gate.py --gate none --eval-mode freeze --gpu 0 --server=sc --dataset imagenet-100 --data /home/gridsan/groups/MAML-Soljacic/imagenet-100 --exp-id=testgate --combine_sep_ckpts --pretrained /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-rotinvdatasetimagenet-100eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier0.3lr-schedulercosineworld-size1rank0/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-roteqdatasetimagenet-100eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier0.3lr-schedulercosineworld-size1rank0/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-baselinedatasetimagenet-100eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier0.3lr-schedulercosineworld-size1rank0/checkpoint.pth


python eval_ensem_gate.py --gate joint --gpu 0 --server=sc --dataset imagenet-100 --exp-id=testgate2 --combine_sep_ckpts --use_default_pretrained
python eval_ensem_gate.py --gate frozen --gpu 0 --server=sc --dataset imagenet-100 --exp-id=testgate --combine_sep_ckpts --use_default_pretrained

python eval_ensem_gate.py --gate joint --gpu 0 --server=sc --dataset imagenet-100 --exp-id=testgate2 --combine_sep_ckpts --use_default_pretrained

for lr in 0.3 1.0 0.1 0.03 0.01 0.001
do
python eval_ensem_gate.py --server=sc --submit --arg_str="--weight_logits --gate=frozen --lr-gate=${lr} --dataset=imagenet-100 --combine_sep_ckpts --use_default_pretrained "
python eval_ensem_gate.py --server=sc --submit --arg_str="--weight_logits --gate=joint --lr-gate=${lr} --lr-classifier=${lr} --dataset=imagenet-100 --combine_sep_ckpts --use_default_pretrained "
done

for lr in 0.3 1.0 0.1 0.01 0.001
do
python eval_ensem_gate.py --server=rumensc --submit --arg_str="--weight_logits --gate_top1 --gate=frozen --lr-gate=${lr} --dataset=imagenet-100 --combine_sep_ckpts --use_default_pretrained "
python eval_ensem_gate.py --server=rumensc --submit --arg_str="--weight_logits --gate_top1 --gate=joint --lr-gate=${lr} --lr-classifier=${lr} --dataset=imagenet-100 --combine_sep_ckpts --use_default_pretrained "
done

for lr in 3.0 0.3 1.0 0.1 0.03
do
python eval_ensem_gate.py --server=rumensc --submit --arg_str="--gate=frozen --cond_x --gate_arch=smallcnn --lr-gate=${lr} --dataset=imagenet-100 --combine_sep_ckpts --use_default_pretrained "
python eval_ensem_gate.py --server=rumensc --submit --arg_str="--gate=joint --cond_x --gate_arch=smallcnn --lr-gate=${lr} --lr-classifier=${lr} --dataset=imagenet-100 --combine_sep_ckpts --use_default_pretrained "
done

for lr in 3.0 0.3 1.0 0.1
do
python eval_ensem_gate.py --server=evansc --submit --arg_str="--gate=frozen --gate_top1 --lr-gate=${lr} --dataset=imagenet-100 --combine_sep_ckpts --use_default_pretrained "
python eval_ensem_gate.py --server=evansc --submit --arg_str="--gate=joint --gate_top1 --lr-gate=${lr} --lr-classifier=${lr} --dataset=imagenet-100 --combine_sep_ckpts --use_default_pretrained "
done

for lr in 0.3
do
python eval_ensem_gate.py --server=evansc --submit --arg_str="--gate=frozen --gate_top1 --cond_x --gate_arch=smallcnn --lr-gate=${lr} --dataset=imagenet-100 --combine_sep_ckpts --use_default_pretrained "
python eval_ensem_gate.py --server=evansc --submit --arg_str="--gate=joint --gate_top1 --cond_x --gate_arch=smallcnn --lr-gate=${lr} --lr-classifier=${lr} --dataset=imagenet-100 --combine_sep_ckpts --use_default_pretrained "
done

for lr in 1.0
do
python eval_ensem_gate.py --server=rumensc --submit --arg_str="--gate=frozen --gate_top1 --cond_x --gate_arch=smallcnn --lr-gate=${lr} --dataset=imagenet-100 --combine_sep_ckpts --use_default_pretrained "
python eval_ensem_gate.py --server=rumensc --submit --arg_str="--gate=joint --gate_top1 --cond_x --gate_arch=smallcnn --lr-gate=${lr} --lr-classifier=${lr} --dataset=imagenet-100 --combine_sep_ckpts --use_default_pretrained "
done

python eval_ensem_gate.py --gate joint --gpu 0 --server=sc --dataset imagenet-100 --exp-id=testgate3 --combine_sep_ckpts --use_default_pretrained

python eval_ensem_gate.py --gate none --eval-mode linear_probe --gpu 0 --server=sc --dataset imagenet-100 --exp-id=testgate3 --combine_sep_ckpts --use_default_pretrained



python eval_ensem_gate.py --lr-scheduler=cosine --gate=frozen --lr-gate=0.001 --dataset=imagenet-100 --combine_sep_ckpts --use_default_pretrained --exp-id=testweightlogits
--- 11/28 ADD lr-scheduler ----

for lr in 0.1 0.01 0.001 0.0001
do
python eval_ensem_gate.py --server=sc --submit --arg_str="--weight_logits --lr-scheduler=cosine --gate=frozen --lr-gate=${lr} --dataset=imagenet-100 --combine_sep_ckpts --use_default_pretrained "
python eval_ensem_gate.py --server=sc --submit --arg_str="--lr-scheduler=cosine --gate=frozen --lr-gate=${lr} --dataset=imagenet-100 --combine_sep_ckpts --use_default_pretrained "
done

for lr in 0.1 0.01 0.001 0.0001
do
python eval_ensem_gate.py --server=rumensc --submit --arg_str="--weight_logits --lr-scheduler=cosine --gate_top1 --gate=frozen --lr-gate=${lr} --dataset=imagenet-100 --combine_sep_ckpts --use_default_pretrained "
python eval_ensem_gate.py --server=rumensc --submit --arg_str="--lr-scheduler=cosine --gate_top1 --gate=frozen --lr-gate=${lr} --lr-classifier=${lr} --dataset=imagenet-100 --combine_sep_ckpts --use_default_pretrained "
done

python eval_ensem_gate.py --lr-scheduler=cosine --gate=frozen --dataset=imagenet-100 --combine_sep_ckpts --use_default_pretrained --exp-id=teststab --lr-gate=0.1
python eval_ensem_gate.py --lr-scheduler=cosine --gate=frozen --dataset=imagenet-100 --combine_sep_ckpts --use_default_pretrained --exp-id=teststab --lr-gate=0.1

for lr in 0.1 0.01 0.001 0.0001
do
python eval_ensem_gate.py --server=rumensc --submit --arg_str="--use_eps --lr-scheduler=cosine --gate=frozen --lr-gate=${lr} --dataset=imagenet-100 --combine_sep_ckpts --use_default_pretrained "
done

python eval_ensem_gate.py --exp-id=testevalensem_base --gate=none --eval-mode=freeze --gpu=0 --server=sc --dataset=imagenet-100 --combine_sep_ckpts --data /home/gridsan/groups/MAML-Soljacic/imagenet-100  --pretrained /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-baselinedatasetimagenet-100eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier0.3lr-schedulercosineworld-size1rank0/checkpoint.pth
python eval_ensem_gate.py --exp-id=testevalensem_rotinv --gate=none --eval-mode=freeze --gpu=0 --server=sc --dataset=imagenet-100 --combine_sep_ckpts --data /home/gridsan/groups/MAML-Soljacic/imagenet-100  --pretrained /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-rotinvdatasetimagenet-100eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier0.3lr-schedulercosineworld-size1rank0/checkpoint.pth
python eval_ensem_gate.py --exp-id=testevalensem_roteq --gate=none --eval-mode=freeze --gpu=0 --server=sc --dataset=imagenet-100 --combine_sep_ckpts --data /home/gridsan/groups/MAML-Soljacic/imagenet-100  --pretrained /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-roteqdatasetimagenet-100eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier0.3lr-schedulercosineworld-size1rank0/checkpoint.pth

python eval_ensem_gate.py --exp-id=testevalensem_roteee_difflr --gate=none --eval-mode=freeze --gpu=0 --server=sc --dataset=imagenet-100 --combine_sep_ckpts --data /home/gridsan/groups/MAML-Soljacic/imagenet-100  --pretrained /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-roteqdatasetimagenet-100eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier0.3lr-schedulercosineworld-size1rank0/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-roteqdatasetimagenet-100eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier1.0lr-schedulercosineworld-size1rank0/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-roteqdatasetimagenet-100eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier5.0lr-schedulercosineworld-size1rank0/checkpoint.pth

python eval_ensem.py --exp-id=testevalensem_roteee_difflr --eval-mode=freeze --gpu=0 --server=sc --dataset=imagenet-100 --combine_sep_ckpts --data /home/gridsan/groups/MAML-Soljacic/imagenet-100  --pretrained /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-roteqdatasetimagenet-100eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier0.3lr-schedulercosineworld-size1rank0/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-roteqdatasetimagenet-100eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier1.0lr-schedulercosineworld-size1rank0/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-roteqdatasetimagenet-100eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier5.0lr-schedulercosineworld-size1rank0/checkpoint.pth


python eval_ensem.py --exp-id=testevalensem_bbb --eval-mode=freeze --gpu=0 --server=sc --dataset=imagenet-100 --combine_sep_ckpts --data /home/gridsan/groups/MAML-Soljacic/imagenet-100  --pretrained /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-baselinedatasetimagenet-100eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier0.3lr-schedulercosineworld-size1rank0/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-baseline2datasetimagenet-100eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier0.3lr-schedulercosineworld-size1rank0/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-baseline3datasetimagenet-100eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier0.3lr-schedulercosineworld-size1rank0/checkpoint.pth
/home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth

python eval_ensem.py --exp-id=imagenet_base --eval-mode=freeze --gpu=0 --server=sc --dataset=imagenet --combine_sep_ckpts --data /home/gridsan/groups/datasets/ImageNet  --pretrained /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth

python eval_ensem.py --exp-id=imagenet_bei --eval-mode=freeze --gpu=0 --server=sc --dataset=imagenet --combine_sep_ckpts --data /home/gridsan/groups/datasets/ImageNet  --pretrained /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth

for lr in 0.00001 0.0005 0.000001
do
python eval_ensem_gate.py --server=sc --submit --arg_str="--weight_logits --lr-scheduler=cosine --gate_top1 --gate=frozen --lr-gate=${lr} --dataset=imagenet-100 --combine_sep_ckpts --use_default_pretrained "
done

python eval_ensem.py --exp-id=testpred --eval-mode=freeze --gpu=0 --server=sc --dataset=imagenet-100 --combine_sep_ckpts --data /home/gridsan/groups/MAML-Soljacic/imagenet-100  --pretrained /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-baselinedatasetimagenet-100eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier0.3lr-schedulercosineworld-size1rank0/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-rotinvdatasetimagenet-100eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier0.3lr-schedulercosineworld-size1rank0/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-roteqdatasetimagenet-100eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier0.3lr-schedulercosineworld-size1rank0/checkpoint.pth

for lr in 1.0 0.1 0.01 0.001 0.0001
do
python eval_ensem_gate.py --server=sc --submit --arg_str="--use_eps --lr-scheduler=cosine --gate=joint --lr-gate=${lr} --dataset=imagenet-100 --combine_sep_ckpts --use_default_pretrained "
done

python eval_ensem.py --ensem_pred=soft_vote_max --exp-id=softvotemax_bei_imagenet --eval-mode=freeze --gpu=0 --server=sc --dataset=imagenet --combine_sep_ckpts --data /home/gridsan/groups/datasets/ImageNet  --pretrained /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth

python eval_ensem_gate.py --server=sc --gpu=0 --use_eps --lr-scheduler=cosine --gate=frozen --lr-gate=0.01 --dataset=imagenet-100 --combine_sep_ckpts --use_default_pretrained --exp-id=testmemax
python eval_ensem_gate.py --gate_arch=smallmlp --server=sc --gpu=0 --use_eps --lr-scheduler=cosine --gate=frozen --lr-gate=0.01 --dataset=imagenet-100 --combine_sep_ckpts --use_default_pretrained --exp-id=testmemax

for lr in 0.01 0.001
do
for lmbd in 0.1 1 0.01
do
python eval_ensem_gate.py --server=sc --submit --arg_str="--me_max --lmbd=${lmbd} --use_eps --lr-scheduler=cosine --gate=frozen --lr-gate=${lr} --dataset=imagenet-100 --combine_sep_ckpts --use_default_pretrained "
done
done

python eval_ensem_gate.py --server=sc --submit --arg_str="--lmbd=0 --use_eps --lr-scheduler=cosine --gate=frozen --lr-gate=0.01 --dataset=imagenet-100 --combine_sep_ckpts --use_default_pretrained "



for lr in 0.1 0.01
do
python eval_ensem_gate.py --server=sc --submit --arg_str="--gate_arch=smallmlp_bn --smallmlp_hd=128 --use_eps --lr-scheduler=cosine --gate=frozen --lr-gate=${lr} --dataset=imagenet-100 --combine_sep_ckpts --use_default_pretrained "
python eval_ensem_gate.py --server=sc --submit --arg_str="--gate_arch=smallmlp_bn --smallmlp_hd=512 --use_eps --lr-scheduler=cosine --gate=frozen --lr-gate=${lr} --dataset=imagenet-100 --combine_sep_ckpts --use_default_pretrained "
done


python eval_ensem_calib.py --server=sc --gpu=0 --eval-mode=temp_scale --lr-temp=0.01 --dataset=imagenet-100 --exp-id=testcalib --combine_sep_ckpts --data=/home/gridsan/groups/MAML-Soljacic/imagenet-100 --pretrained=/home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-baselinedatasetimagenet-100eval-modelinear_probebatch-size256lr-backbone0.0lr-classifier0.3lr-schedulercosineworld-size1rank0/checkpoint.pth


python eval_ensem_calib.py --server=sc --gpu=0 --eval-mode=linear_probe --lr-classifier=0.1 --dataset=imagenet-100 --use_smaller_split --exp-id=testlp --data=/home/gridsan/groups/MAML-Soljacic/imagenet-100 --pretrained=/home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth

for lr in 0.1 0.3 0.5 1.0 5.0
do
python eval_ensem_calib.py --server=rumensc --submit --arg_str="--combine_sep_ckpts --eval-mode=linear_probe --lp_100_on_full=base --lr-classifier=${lr} --dataset=imagenet-100 --use_smaller_split --val_perc=20 --lr-scheduler=cosine "
done

for lr in 0.1 0.3 0.5 1.0 5.0
do
python eval_ensem_calib.py --server=rumensc --submit --arg_str="--combine_sep_ckpts --eval-mode=linear_probe --lp_100_on_full=base --lr-classifier=${lr} --dataset=imagenet-100 --lr-scheduler=cosine "
done


python eval_ensem.py --exp-id=imagenet_evalon100_bei --eval_subset100 --eval-mode=freeze --gpu=0 --server=sc --dataset=imagenet --combine_sep_ckpts --data /home/gridsan/groups/datasets/ImageNet  --pretrained /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth
python eval_ensem.py --exp-id=imagenet_evalon100_base --eval_subset100 --eval-mode=freeze --gpu=0 --server=sc --dataset=imagenet --combine_sep_ckpts --data /home/gridsan/groups/datasets/ImageNet  --pretrained /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth
python eval_ensem.py --exp-id=imagenet_evalon100_eq --eval_subset100 --eval-mode=freeze --gpu=0 --server=sc --dataset=imagenet --combine_sep_ckpts --data /home/gridsan/groups/datasets/ImageNet  --pretrained /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth
python eval_ensem.py --exp-id=imagenet_evalon100_inv --eval_subset100 --eval-mode=freeze --gpu=0 --server=sc --dataset=imagenet --combine_sep_ckpts --data /home/gridsan/groups/datasets/ImageNet  --pretrained /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth


for lr in 0.001
do
for lmbd in 0.05 0.01 0.005
do
python eval_ensem_gate.py --server=sc --submit --arg_str="--me_max --lmbd=${lmbd} --use_eps --lr-scheduler=cosine --gate=frozen --lr-gate=${lr} --dataset=imagenet-100 --combine_sep_ckpts --use_default_pretrained "
done
done


# mlp gate on train
python eval_ensem_gate.py --exp-id=in_mlpselect_evalon100_bei_trainset --eval_on_train --eval_subset100 --gate_arch=mlp_selector --server=sc --gpu=0 --gate=all_frozen --eval-mode=freeze --dataset=imagenet --combine_sep_ckpts --data /home/gridsan/groups/datasets/ImageNet  --pretrained /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth --gate_pretrained /home/gridsan/groups/MAML-Soljacic/selector_checkpoints/in-100_mlp_classwise/model_best.pth.tar
# rn18 gate on train
python eval_ensem_gate.py --exp-id=in_rn18select_evalon100_bei_trainset --eval_on_train --eval_subset100 --gate_arch=rn18_selector --cond_x --server=sc --gpu=0 --gate=all_frozen --eval-mode=freeze --dataset=imagenet --combine_sep_ckpts --data /home/gridsan/groups/datasets/ImageNet  --pretrained /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth --gate_pretrained /home/gridsan/groups/MAML-Soljacic/selector_checkpoints/in-100_resnet18_classwise/model_best.pth.tar

# mlp gate on eval
python eval_ensem_gate.py --exp-id=in_mlpselect_evalon100_bei --eval_subset100 --gate_arch=mlp_selector --server=sc --gpu=0 --gate=all_frozen --eval-mode=freeze --dataset=imagenet --combine_sep_ckpts --data /home/gridsan/groups/datasets/ImageNet  --pretrained /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth --gate_pretrained /home/gridsan/groups/MAML-Soljacic/selector_checkpoints/in-100_mlp_classwise/model_best.pth.tar
# rn18 gate on eval
python eval_ensem_gate.py --exp-id=in_rn18select_evalon100_bei --eval_subset100 --gate_arch=rn18_selector --cond_x --server=sc --gpu=0 --gate=all_frozen --eval-mode=freeze --dataset=imagenet --combine_sep_ckpts --data /home/gridsan/groups/datasets/ImageNet  --pretrained /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth --gate_pretrained /home/gridsan/groups/MAML-Soljacic/selector_checkpoints/in-100_resnet18_classwise/model_best.pth.tar

# mlp gate on train + gate top 1
python eval_ensem_gate.py --exp-id=in_mlpselect_top1_evalon100_bei_trainset --gate_top1 --eval_on_train --eval_subset100 --gate_arch=mlp_selector --server=sc --gpu=0 --gate=all_frozen --eval-mode=freeze --dataset=imagenet --combine_sep_ckpts --data /home/gridsan/groups/datasets/ImageNet  --pretrained /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth --gate_pretrained /home/gridsan/groups/MAML-Soljacic/selector_checkpoints/in-100_mlp_classwise/model_best.pth.tar
# rn18 gate on train + gate top 1
python eval_ensem_gate.py --exp-id=in_rn18select_top1_evalon100_bei_trainset --gate_top1 --eval_on_train --eval_subset100 --gate_arch=rn18_selector --cond_x --server=sc --gpu=0 --gate=all_frozen --eval-mode=freeze --dataset=imagenet --combine_sep_ckpts --data /home/gridsan/groups/datasets/ImageNet  --pretrained /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth --gate_pretrained /home/gridsan/groups/MAML-Soljacic/selector_checkpoints/in-100_resnet18_classwise/model_best.pth.tar

# mlp gate on eval + gate top 1
python eval_ensem_gate.py --exp-id=in_mlpselect_top1_evalon100_bei --gate_top1 --eval_subset100 --gate_arch=mlp_selector --server=sc --gpu=0 --gate=all_frozen --eval-mode=freeze --dataset=imagenet --combine_sep_ckpts --data /home/gridsan/groups/datasets/ImageNet  --pretrained /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth --gate_pretrained /home/gridsan/groups/MAML-Soljacic/selector_checkpoints/in-100_mlp_classwise/model_best.pth.tar
# rn18 gate on eval + gate top 1
python eval_ensem_gate.py --exp-id=in_rn18select_top1_evalon100_bei --gate_top1 --eval_subset100 --gate_arch=rn18_selector --cond_x --server=sc --gpu=0 --gate=all_frozen --eval-mode=freeze --dataset=imagenet --combine_sep_ckpts --data /home/gridsan/groups/datasets/ImageNet  --pretrained /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth --gate_pretrained /home/gridsan/groups/MAML-Soljacic/selector_checkpoints/in-100_resnet18_classwise/model_best.pth.tar

python eval_ensem_gate.py --exp-id=in_mlpselect_evalon100_bei --eval_subset100 --gate_arch=mlp_selector --server=sc --gpu=0 --gate=all_frozen --eval-mode=freeze --dataset=imagenet --combine_sep_ckpts --data /home/gridsan/groups/datasets/ImageNet  --pretrained /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth --gate_pretrained /home/gridsan/groups/MAML-Soljacic/selector_checkpoints/in-100_mlp_classwise/model_best.pth.tar; python eval_ensem_gate.py --exp-id=in_mlpselect_top1_evalon100_bei --gate_top1 --eval_subset100 --gate_arch=mlp_selector --server=sc --gpu=0 --gate=all_frozen --eval-mode=freeze --dataset=imagenet --combine_sep_ckpts --data /home/gridsan/groups/datasets/ImageNet  --pretrained /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth --gate_pretrained /home/gridsan/groups/MAML-Soljacic/selector_checkpoints/in-100_mlp_classwise/model_best.pth.tar; python eval_ensem_gate.py --exp-id=in_mlpselect_evalon100_bei_trainset --eval_on_train --eval_subset100 --gate_arch=mlp_selector --server=sc --gpu=0 --gate=all_frozen --eval-mode=freeze --dataset=imagenet --combine_sep_ckpts --data /home/gridsan/groups/datasets/ImageNet  --pretrained /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth --gate_pretrained /home/gridsan/groups/MAML-Soljacic/selector_checkpoints/in-100_mlp_classwise/model_best.pth.tar; python eval_ensem_gate.py --exp-id=in_mlpselect_top1_evalon100_bei_trainset --gate_top1 --eval_on_train --eval_subset100 --gate_arch=mlp_selector --server=sc --gpu=0 --gate=all_frozen --eval-mode=freeze --dataset=imagenet --combine_sep_ckpts --data /home/gridsan/groups/datasets/ImageNet  --pretrained /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth --gate_pretrained /home/gridsan/groups/MAML-Soljacic/selector_checkpoints/in-100_mlp_classwise/model_best.pth.tar

for lr in 0.5 1.0 2.0
do
python eval_ensem_calib.py --server=rumensc --submit --arg_str="--combine_sep_ckpts --eval-mode=linear_probe --lp_100_on_full=base --lr-classifier=${lr} --dataset=imagenet-100 --use_smaller_split --val_perc=20 --lr-scheduler=cosine "
done

for lr in 1.0
do
python eval_ensem_calib.py --server=rumensc --submit --arg_str="--combine_sep_ckpts --eval-mode=linear_probe --lp_100_on_full=eq --lr-classifier=${lr} --dataset=imagenet-100 --use_smaller_split --val_perc=20 --lr-scheduler=cosine "
python eval_ensem_calib.py --server=rumensc --submit --arg_str="--combine_sep_ckpts --eval-mode=linear_probe --lp_100_on_full=eq --lr-classifier=${lr} --dataset=imagenet-100 --lr-scheduler=cosine "
python eval_ensem_calib.py --server=rumensc --submit --arg_str="--combine_sep_ckpts --eval-mode=linear_probe --lp_100_on_full=inv --lr-classifier=${lr} --dataset=imagenet-100 --use_smaller_split --val_perc=20 --lr-scheduler=cosine "
python eval_ensem_calib.py --server=rumensc --submit --arg_str="--combine_sep_ckpts --eval-mode=linear_probe --lp_100_on_full=inv --lr-classifier=${lr} --dataset=imagenet-100 --lr-scheduler=cosine "
done


for lr in 0.001
do
for lmbd in 0.05 0.001 0.005
do
python eval_ensem_gate.py --server=sc --submit --arg_str="--me_max --lmbd=${lmbd} --use_eps --lr-scheduler=cosine --gate=frozen --lr-gate=${lr} --dataset=imagenet-100 --combine_sep_ckpts --use_default_pretrained "
done
done

for lr in 0.01 0.001
do
for lmbd in 0.01 0.001
do
python eval_ensem_gate.py --server=sc --submit --arg_str="--me_max --lmbd=${lmbd} --use_eps --lr-scheduler=cosine --gate=frozen --lr-gate=${lr} --eval_subset100 --dataset=imagenet --combine_sep_ckpts --use_default_pretrained "
done
done

for lr in 0.1 0.01 0.001
do
for lmbd in 0.01
do
python eval_ensem_gate.py --server=sc --submit --arg_str="--gate_arch=smallmlp --smallmlp_hd=256 --me_max --lmbd=${lmbd} --use_eps --lr-scheduler=cosine --gate=frozen --lr-gate=${lr} --eval_subset100 --dataset=imagenet --combine_sep_ckpts --use_default_pretrained "
done
done
# python eval_ensem_gate.py --server=rumensc --gpu=0 --exp-id=testsmallmlp --gate_arch=smallmlp --smallmlp_hd=128 --me_max --lmbd=0.1 --use_eps --lr-scheduler=cosine --gate=frozen --lr-gate=0.1 --eval_subset100 --dataset=imagenet --combine_sep_ckpts --use_default_pretrained

for lr in 0.1 0.01 0.001
do
for lmbd in 0.001
do
python eval_ensem_gate.py --server=sc --submit --arg_str="--gate_arch=smallmlp --smallmlp_hd=256 --me_max --lmbd=${lmbd} --use_eps --lr-scheduler=cosine --gate=frozen --lr-gate=${lr} --eval_subset100 --dataset=imagenet --combine_sep_ckpts --use_default_pretrained "
done
done

python eval_ensem_gate.py --exp-id=testmlpselect --eval_subset100 --gate_arch=mlp_selector --server=sc --gpu=0 --gate=all_frozen --eval-mode=freeze --dataset=imagenet --combine_sep_ckpts --data /home/gridsan/groups/datasets/ImageNet  --pretrained /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth --gate_pretrained /home/gridsan/groups/MAML-Soljacic/selector_checkpoints/in-100_mlp_classwise/model_best.pth.tar



python eval_ensem_calib.py --server=sc --gpu=0 --exp-id=test20psplit --combine_sep_ckpts --eval-mode=linear_probe --lp_100_on_full=eq --lr-classifier=1.0 --dataset=imagenet-100 --use_smaller_split --val_perc=20 --lr-scheduler=cosine


for lr in 1.0
do
python eval_ensem_calib.py --server=rumensc --submit --arg_str="--combine_sep_ckpts --eval-mode=linear_probe --lp_100_on_full=eq --lr-classifier=${lr} --dataset=imagenet-100 --use_smaller_split --val_perc=20 --lr-scheduler=cosine "
python eval_ensem_calib.py --server=rumensc --submit --arg_str="--combine_sep_ckpts --eval-mode=linear_probe --lp_100_on_full=inv --lr-classifier=${lr} --dataset=imagenet-100 --use_smaller_split --val_perc=20 --lr-scheduler=cosine "
python eval_ensem_calib.py --server=rumensc --submit --arg_str="--combine_sep_ckpts --eval-mode=linear_probe --lp_100_on_full=base --lr-classifier=${lr} --dataset=imagenet-100 --use_smaller_split --val_perc=20 --lr-scheduler=cosine "
done

for lr in 0.01 0.001
do
python eval_ensem_gate.py --server=sc --submit --arg_str="--lmbd=0 --use_eps --lr-scheduler=cosine --gate=frozen --lr-gate=${lr} --eval_subset100 --dataset=imagenet --combine_sep_ckpts --use_default_pretrained "
done



python eval_ensem.py --exp-id=imagenet_bei_trainset --eval_on_train --eval-mode=freeze --gpu=0 --server=sc --dataset=imagenet --combine_sep_ckpts --data /home/gridsan/groups/datasets/ImageNet  --pretrained /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth
python eval_ensem.py --exp-id=imagenet_base_trainset --eval_on_train --eval-mode=freeze --gpu=0 --server=sc --dataset=imagenet --combine_sep_ckpts --data /home/gridsan/groups/datasets/ImageNet  --pretrained /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth
python eval_ensem.py --exp-id=imagenet_eq_trainset --eval_on_train --eval-mode=freeze --gpu=0 --server=sc --dataset=imagenet --combine_sep_ckpts --data /home/gridsan/groups/datasets/ImageNet  --pretrained /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth
python eval_ensem.py --exp-id=imagenet_inv_trainset --eval_on_train --eval-mode=freeze --gpu=0 --server=sc --dataset=imagenet --combine_sep_ckpts --data /home/gridsan/groups/datasets/ImageNet  --pretrained /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth




python eval_ensem.py --exp-id=imagenet_eval100_bei_trainset --eval_subset100 --eval_on_train --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth

python eval_ensem.py --exp-id=imagenet_bei_trainset --eval_on_train --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /home/gridsan/groups/datasets/ImageNet  --pretrained /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth
python eval_ensem.py --exp-id=imagenet_eq_trainset --eval_on_train --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth; python eval_ensem.py --exp-id=imagenet_inv_trainset --eval_on_train --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth

python eval_ensem.py --exp-id=imagenet_base_trainset --eval_on_train --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth


python eval_ensem.py --exp-id=imagenet_bei --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /home/gridsan/groups/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth; python eval_ensem.py --exp-id=imagenet_eq --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth; python eval_ensem.py --exp-id=imagenet_inv --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth; python eval_ensem.py --exp-id=imagenet_base --eval_on_train --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth


python eval_ensem.py --exp-id=imagenet_base --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth
python eval_ensem.py --exp-id=imagenet_base2 --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/checkpoints/10-16-simclr-rot-0.0-800ep-resnet50.pth



python eval_ensem.py --exp-id=imagenet_bei_trainset2 --eval_on_train --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth; python eval_ensem.py --exp-id=imagenet_eq_trainset2 --eval_on_train --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth; python eval_ensem.py --exp-id=imagenet_inv_trainset --eval_on_train --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth; python eval_ensem.py --exp-id=imagenet_base_trainset2 --eval_on_train --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth




python eval_ensem_bei_labels.py --exp-id=imagenet_beilabels --use_bei_labels --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth

python eval_ensem_bei_labels.py --exp-id=imagenet_uniqbei --uniq_bei --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth

python eval_ensem_bei_labels.py --exp-id=imagenet_nonuniqbei --non_uniq_bei --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth

python eval_ensem_bei_selectclasses.py --exp-id=imagenet_sel --select_classes=best_bie --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth

python eval_ensem_bei_selectclasses.py --exp-id=imagenet_sel_bi --select_classes=best_bi --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth

python eval_ensem_bei_selectclasses.py --exp-id=imagenet_sel_be --select_classes=best_be --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth;
python eval_ensem_bei_selectclasses.py --exp-id=imagenet_sel_ie --select_classes=best_ie --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth;
python eval_ensem_bei_selectclasses.py --exp-id=imagenet_sel_b --select_classes=best_b --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth;
python eval_ensem_bei_selectclasses.py --exp-id=imagenet_sel_i --select_classes=best_i --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth;
python eval_ensem_bei_selectclasses.py --exp-id=imagenet_sel_e --select_classes=best_e --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth

python eval_ensem_bei_selectclasses.py --exp-id=imagenet_sel_e_model_b --select_classes=best_e --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth; python eval_ensem_bei_selectclasses.py --exp-id=imagenet_sel_e_model_i --select_classes=best_e --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth

python eval_ensem_bei_selectclasses.py --exp-id=imagenet_sel_b_model_e --select_classes=best_b --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth; python eval_ensem_bei_selectclasses.py --exp-id=imagenet_sel_i_model_e --select_classes=best_i --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth


python eval_ensem_bei_selectclasses.py --exp-id=imagenet_sel_e_model_e69 --select_classes=best_e --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-roteq-seed69-ep800-lp-lr0.3-cosine/checkpoint.pth


python eval_ensem.py --exp-id=imagenet_eq_seed69 --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-roteq-seed69-ep800-lp-lr0.3-cosine/checkpoint.pth


python eval_ensem_bei_selectclasses.py --exp-id=imagenet_sel_b_model_e69 --select_classes=best_b --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-roteq-seed69-ep800-lp-lr0.3-cosine/checkpoint.pth; python eval_ensem_bei_selectclasses.py --exp-id=imagenet_sel_i_model_e69 --select_classes=best_i --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-roteq-seed69-ep800-lp-lr0.3-cosine/checkpoint.pth



for lr in 1.0
do
python eval_ensem_calib.py --server=rumensc --submit --add_prefix=rr --arg_str="--combine_sep_ckpts --eval-mode=linear_probe --lp_100_on_full=base --lr-classifier=${lr} --dataset=imagenet-100 --lr-scheduler=cosine "
done



python eval_ensem_calib.py --server=aimos --gpu=0 --combine_sep_ckpts --eval-mode=linear_probe --lr-classifier=0.1 --dataset=imagenet --use_smaller_split --val_perc=20 --lr-scheduler=cosine --data /gpfs/u/locker/200/CADS/datasets/ImageNet/ --exp-id testlpcal --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth



python eval_ensem.py --exp-id=testpra --eval_on_train --eval-mode=freeze --gpu=0 --server=sc --dataset=imagenet --combine_sep_ckpts --data /home/gridsan/groups/datasets/ImageNet  --pretrained /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth /home/gridsan/groups/MAML-Soljacic/lp_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth


python eval_ensem.py --exp-id=imagenet_eq_seed69 --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-roteq-seed69-ep800-lp-lr0.3-cosine/checkpoint.pth

python eval_ensem.py --exp-id=testpra2 --eval-mode=freeze --gpu=0 --server=sc --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-roteq-seed69-ep800-lp-lr0.3-cosine/checkpoint.pth


python eval_ensem.py --exp-id=imagenet_bei2 --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth; python eval_ensem.py --exp-id=imagenet_eq2 --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth; python eval_ensem.py --exp-id=imagenet_inv2 --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth; python eval_ensem.py --exp-id=imagenet_base2 --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth


python eval_ensem.py --server=aimos --gpu=0 --combine_sep_ckpts --eval-mode=linear_probe --lr-classifier=0.3 --dataset=imagenet --lr-scheduler=cosine --data /gpfs/u/locker/200/CADS/datasets/ImageNet/ --exp-id lp_eq_seed69 --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_pt_checkpoints/simclr-roteq-seed69-IN1k-800e-resnet50.pth

python eval_ensem.py --server=aimos --gpu=0 --combine_sep_ckpts --eval-mode=freeze --dataset=imagenet --lr-scheduler=cosine --data /gpfs/u/locker/200/CADS/datasets/ImageNet/ --exp-id TESTEVAL_lp_eq_seed69 --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/checkpoints/lp_eq_seed69/checkpoint.pth


python eval_ensem_calib.py --server=aimos --gpu=0 --combine_sep_ckpts --eval-mode=linear_probe --lr-classifier=0.1 --dataset=imagenet --use_smaller_split --val_perc=20 --lr-scheduler=cosine --data /gpfs/u/locker/200/CADS/datasets/ImageNet/ --exp-id testlpcal --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth


python eval_ensem_calib.py --server=aimos --gpu=0 --eval-mode=temp_scale --lr-temp=0.01 --dataset=imagenet --exp-id=testcalib --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet/ --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth

for lr in 0.001
do
for lmbd in 0.005 0.03
do
python eval_ensem_gate.py --server=sc --submit --arg_str="--gate_arch=smallmlp --smallmlp_hd=256 --me_max --lmbd=${lmbd} --use_eps --lr-scheduler=cosine --gate=frozen --lr-gate=${lr} --eval_subset100 --dataset=imagenet --combine_sep_ckpts --use_default_pretrained "
done
done

for lr in 0.001 0.0005
do
for lmbd in 0.001 0.01
do
python eval_ensem_gate.py --server=sc --submit --arg_str="--gate_arch=smallmlp --smallmlp_hd=128 --me_max --lmbd=${lmbd} --use_eps --lr-scheduler=cosine --gate=frozen --lr-gate=${lr} --eval_subset100 --dataset=imagenet --combine_sep_ckpts --use_default_pretrained "
done
done

for lr in 0.001 0.01
do
for lmbd in 0.001 0.01
do
python eval_ensem_gate.py --server=aimos --submit --arg_str="--gate_arch=smallmlp --smallmlp_hd=256 --me_max --lmbd=${lmbd} --use_eps --lr-scheduler=cosine --gate=frozen --lr-gate=${lr} --dataset=imagenet --combine_sep_ckpts --use_default_pretrained "
done
done

python eval_ensem_gate.py --server=aimos --gate_arch=smallmlp --smallmlp_hd=128 --me_max --lmbd=0.01 --use_eps --lr-scheduler=cosine --gate=frozen --lr-gate=0.001 --dataset=imagenet --combine_sep_ckpts --use_default_pretrained

python eval_ensem_calib.py --exp-id=ts_base0.3_80 --server=aimos --gpu=0 --eval-mode=temp_scale --lr-temp=0.01 --dataset=imagenet  --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet/ --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp80perc_baseline_cos_lr0.3_bs256/checkpoint.pth

python eval_ensem_calib.py --server=aimos --gpu=0 --combine_sep_ckpts --eval-mode=linear_probe --lr-classifier=0.3 --dataset=imagenet --use_smaller_split --val_perc=20 --lr-scheduler=cosine --data /gpfs/u/locker/200/CADS/datasets/ImageNet/ --exp-id testlpcal90_2 --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth


python eval_ensem.py --exp-id=imagenet_eq_seed42 --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp_eq_seed42_cos_lr0.3_bs256/checkpoint.pth

for lr in 1 3 10
do
for miter in 50 100
do
python eval_ensem_calib.py --exp-id=ts_base0.3_80 --server=aimos --gpu=0 --eval-mode=temp_scale --lr-temp=${lr} --max_iter=${miter} --dataset=imagenet  --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet/ --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp80perc_baseline_cos_lr0.3_bs256/checkpoint.pth
done
done

python eval_ensem.py --exp-id=imagenet_eee --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/lp_eq_seed42_cos_lr0.3_bs256/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/lp_eq_seed69_cos_lr0.3_bs256/checkpoint.pth


python eval_ensem.py --exp-id=test_ft --lr-classifier=0.01 --lr-backbone=0.01 --lr-scheduler=cosine --eval-mode=finetune --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/lp_eq_seed42_cos_lr0.3_bs256/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/lp_eq_seed69_cos_lr0.3_bs256/checkpoint.pth

python eval_ensem.py --exp-id=test_ft --lr-classifier=0.01 --lr-backbone=0.01 --lr-scheduler=cosine --eval-mode=finetune --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth

python eval_ensem.py --exp-id=imagenet_bei_80perc --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp80perc_baseline_cos_lr0.3_bs256/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp80perc_rotinv_cos_lr0.3_bs256/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp80perc_roteq_cos_lr0.3_bs256/checkpoint.pth

python eval_ensem.py --exp-id=imagenet_bei_on20perc --use_smaller_split_val --eval-mode=freeze --eval_on_train --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp80perc_baseline_cos_lr0.3_bs256/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp80perc_rotinv_cos_lr0.3_bs256/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp80perc_roteq_cos_lr0.3_bs256/checkpoint.pth; python eval_ensem.py --exp-id=imagenet_bei_on80perc --use_smaller_split --eval-mode=freeze --eval_on_train --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp80perc_baseline_cos_lr0.3_bs256/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp80perc_rotinv_cos_lr0.3_bs256/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp80perc_roteq_cos_lr0.3_bs256/checkpoint.pth

python eval_ensem.py --exp-id=imagenet_base_on20perc --use_smaller_split_val --eval-mode=freeze --eval_on_train --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp80perc_baseline_cos_lr0.3_bs256/checkpoint.pth; python eval_ensem.py --exp-id=imagenet_base_on80perc --use_smaller_split --eval-mode=freeze --eval_on_train --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp80perc_baseline_cos_lr0.3_bs256/checkpoint.pth; python eval_ensem.py --exp-id=imagenet_eq_on20perc --use_smaller_split_val --eval-mode=freeze --eval_on_train --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp80perc_roteq_cos_lr0.3_bs256/checkpoint.pth; python eval_ensem.py --exp-id=imagenet_eq_on80perc --use_smaller_split --eval-mode=freeze --eval_on_train --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp80perc_roteq_cos_lr0.3_bs256/checkpoint.pth; python eval_ensem.py --exp-id=imagenet_inv_on20perc --use_smaller_split_val --eval-mode=freeze --eval_on_train --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp80perc_rotinv_cos_lr0.3_bs256/checkpoint.pth; python eval_ensem.py --exp-id=imagenet_inv_on80perc --use_smaller_split --eval-mode=freeze --eval_on_train --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp80perc_rotinv_cos_lr0.3_bs256/checkpoint.pth;


python eval_ensem_gate.py --exp-id=gate200 --server=aimos --gpu=0 --gate_arch=smallmlp --smallmlp_hd=256 --me_max --lmbd=0.01 --use_eps --lr-scheduler=cosine --gate=frozen --lr-gate=0.01 --eval_var_subset=200 --dataset=imagenet --combine_sep_ckpts --use_default_pretrained


python eval_ensem_gate.py --submit --server=aimos --arg_str="--gate_arch=smallmlp --smallmlp_hd=256 --me_max --lmbd=0.01 --use_eps --lr-scheduler=cosine --gate=frozen --lr-gate=0.01 --eval_subset100 --dataset=imagenet --combine_sep_ckpts --use_default_pretrained "
python eval_ensem_gate.py --submit --server=aimos --arg_str="--gate_arch=smallmlp --smallmlp_hd=256 --me_max --lmbd=0.01 --use_eps --lr-scheduler=cosine --gate=frozen --lr-gate=0.001 --eval_subset100 --dataset=imagenet --combine_sep_ckpts --use_default_pretrained "
python eval_ensem_gate.py --submit --server=aimos --arg_str="--gate_arch=smallmlp --smallmlp_hd=256 --me_max --lmbd=0 --use_eps --lr-scheduler=cosine --gate=frozen --lr-gate=0.01 --eval_subset100 --dataset=imagenet --combine_sep_ckpts --use_default_pretrained "; python eval_ensem_gate.py --submit --server=aimos --arg_str="--gate_arch=smallmlp --smallmlp_hd=256 --me_max --lmbd=0 --use_eps --lr-scheduler=cosine --gate=frozen --lr-gate=0.001 --eval_subset100 --dataset=imagenet --combine_sep_ckpts --use_default_pretrained "

for ss in 200 500
do
for lr in 0.001 0.01 0.0001
do
python eval_ensem_gate.py --submit --server=aimos --arg_str="--gate_arch=smallmlp --smallmlp_hd=256 --me_max --lmbd=0.0 --use_eps --lr-scheduler=cosine --gate=frozen --lr-gate=${lr} --eval_var_subset=${ss} --dataset=imagenet --combine_sep_ckpts --use_default_pretrained "
python eval_ensem_gate.py --submit --server=aimos --arg_str="--gate_arch=smallmlp --smallmlp_hd=512 --me_max --lmbd=0.0 --use_eps --lr-scheduler=cosine --gate=frozen --lr-gate=${lr} --eval_var_subset=${ss} --dataset=imagenet --combine_sep_ckpts --use_default_pretrained "
done
done


python eval_ensem.py --exp-id=imagenet_bei_ss100 --eval_subset100 --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth

for ss in 100
do
python eval_ensem.py --exp-id=imagenet_bei_am${ss} --eval_var_subset=${ss} --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth
done


python eval_ensem_calib.py --exp-id=ts_base0.3_95_2 --val_perc=5 --server=aimos --gpu=0 --eval-mode=temp_scale --lr-temp=0.01 --dataset=imagenet  --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet/ --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp95perc_baseline_cos_lr0.3_bs256/checkpoint.pth

python eval_ensem.py --exp-id=imagenet_bei_90perc_ontest --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp90perc_baseline_cos_lr0.3_bs256/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp90perc_rotinv_cos_lr0.3_bs256/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp90perc_roteq_cos_lr0.3_bs256/checkpoint.pth; python eval_ensem.py --exp-id=imagenet_bei_90perc_onval --val_perc=10 --use_smaller_split_val --eval_on_train --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp90perc_baseline_cos_lr0.3_bs256/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp90perc_rotinv_cos_lr0.3_bs256/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp90perc_roteq_cos_lr0.3_bs256/checkpoint.pth; python eval_ensem.py --exp-id=imagenet_bei_95perc_ontest --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp95perc_baseline_cos_lr0.3_bs256/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp95perc_rotinv_cos_lr0.3_bs256/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp95perc_roteq_cos_lr0.3_bs256/checkpoint.pth; python eval_ensem.py --exp-id=imagenet_bei_95perc_onval --val_perc=5 --use_smaller_split_val --eval_on_train --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp95perc_baseline_cos_lr0.3_bs256/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp95perc_rotinv_cos_lr0.3_bs256/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp95perc_roteq_cos_lr0.3_bs256/checkpoint.pth

for lr in 0.01 0.1 0.03 0.03 0.001
do
python eval_ensem_calib.py --exp-id=ts_base0.3_95 --val_perc=5 --server=aimos --gpu=0 --eval-mode=temp_scale --lr-temp=${lr} --dataset=imagenet  --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet/ --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp95perc_baseline_cos_lr0.3_bs256/checkpoint.pth
done


python eval_ensem.py --exp-id=imagenet_eeeee --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp_eq_seed42_cos_lr0.3_bs256/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp_eq_seed69_cos_lr0.3_bs256/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp_eq_seed31_cos_lr0.3_bs256/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp_eq_seed24_cos_lr0.3_bs256/checkpoint_best.pth

python eval_ensem.py --exp-id=imagenet_beiee --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-baseline-ep800-lp-lr0.3-cosine/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-roteq-ep800-lp-lr0.3-cosine/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_checkpoints/simclr-rotinv-ep800-lp-lr0.3-cosine/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp_eq_seed42_cos_lr0.3_bs256/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp_eq_seed69_cos_lr0.3_bs256/checkpoint.pth


python eval_ensem_gate.py --exp-id=testorigate --gpu=0 --server=aimos --gate_arch=smallmlp --smallmlp_hd=1024 --me_max --lmbd=0.0 --use_eps --lr-scheduler=cosine --gate=frozen --lr-gate=0.0001 --dataset=imagenet --combine_sep_ckpts --use_default_pretrained

python eval_ensem_gate.py --exp-id=testvit --gpu=0 --server=aimos --gate_arch=vit_tiny --vit_patch_size=2048 --me_max --lmbd=0.0 --use_eps --lr-scheduler=cosine --gate=frozen --lr-gate=0.01 --dataset=imagenet --combine_sep_ckpts --use_default_pretrained

python eval_ensem_gate.py --exp-id=testorigate95 --gpu=0 --server=aimos --gate_arch=smallmlp --smallmlp_hd=1024 --me_max --lmbd=0.0 --val_perc=5 --use_smaller_split_val --use_eps --lr-scheduler=cosine --gate=frozen --lr-gate=0.0001 --dataset=imagenet --combine_sep_ckpts --pretrained  /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp95perc_baseline_cos_lr0.3_bs256/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp95perc_rotinv_cos_lr0.3_bs256/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp95perc_roteq_cos_lr0.3_bs256/checkpoint.pth --data /gpfs/u/locker/200/CADS/datasets/ImageNet


/gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp95perc_rotinv_cos_lr0.3_bs256/checkpoint.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp95perc_roteq_cos_lr0.3_bs256/checkpoint.pth


python eval_ensem.py --exp-id=imagenet_base_95perc_ontest --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp95perc_baseline_cos_lr0.3_bs256/checkpoint.pth; \
python eval_ensem.py --exp-id=imagenet_eq_95perc_ontest --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp95perc_roteq_cos_lr0.3_bs256/checkpoint.pth; \
python eval_ensem.py --exp-id=imagenet_inv_95perc_ontest --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp95perc_rotinv_cos_lr0.3_bs256/checkpoint.pth; \
python eval_ensem.py --exp-id=imagenet_base_95perc_onval --val_perc=5 --use_smaller_split_val --eval_on_train --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp95perc_baseline_cos_lr0.3_bs256/checkpoint.pth; \
python eval_ensem.py --exp-id=imagenet_eq_95perc_onval --val_perc=5 --use_smaller_split_val --eval_on_train --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp95perc_roteq_cos_lr0.3_bs256/checkpoint.pth; \
python eval_ensem.py --exp-id=imagenet_inv_95perc_onval --val_perc=5 --use_smaller_split_val --eval_on_train --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp95perc_rotinv_cos_lr0.3_bs256/checkpoint.pth; \

python eval_ensem_gate.py --exp-id=testgate_condx50 --cond_x --gpu=0 --server=aimos --gate_arch=resnet50 --me_max --lmbd=0.0 --use_eps --lr-scheduler=cosine --gate=frozen --lr-gate=0.0001 --dataset=imagenet --combine_sep_ckpts --use_default_pretrained
python eval_ensem_gate.py --exp-id=testgate_att --batch-size=13 --cond_x --gpu=0 --server=aimos --gate_arch=resnet50_scaledatt --me_max --lmbd=0.0 --use_eps --lr-scheduler=cosine --gate=frozen --lr-gate=0.0001 --dataset=imagenet --combine_sep_ckpts --use_default_pretrained

python eval_ensem.py --exp-id=imagenet_eee_lpseeds --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp_eq_seed69_cos_lr0.3_bs256_lpseed1/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp_eq_seed69_cos_lr0.3_bs256_lpseed2/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/lp_eq_seed69_cos_lr0.3_bs256_lpseed3/checkpoint_best.pth;

python eval_ensem.py --exp-id=imagenet_bei_ft --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_baseR_cos_lr0.003_bs256/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_inv_cos_lr0.003_bs256/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_eqR_cos_lr0.003_bs256/checkpoint_best.pth;
python eval_ensem.py --exp-id=imagenet_base_ft --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_baseR_cos_lr0.003_bs256/checkpoint_best.pth; \
python eval_ensem.py --exp-id=imagenet_inv_ft --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_inv_cos_lr0.003_bs256/checkpoint_best.pth; \
python eval_ensem.py --exp-id=imagenet_eqR_ft --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_eqR_cos_lr0.003_bs256/checkpoint_best.pth;

python eval_ensem.py --exp-id=imagenet_bei_ft95p --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft95perc_baseR_cos_lr0.003_bs256/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft95perc_inv_cos_lr0.003_bs256/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft95perc_eqR_cos_lr0.003_bs256/checkpoint_best.pth; \
python eval_ensem.py --exp-id=imagenet_bei_ft95p_onval --val_perc=5 --use_smaller_split_val --eval_on_train --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft95perc_baseR_cos_lr0.003_bs256/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft95perc_inv_cos_lr0.003_bs256/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft95perc_eqR_cos_lr0.003_bs256/checkpoint_best.pth; \
python eval_ensem.py --exp-id=imagenet_base_ft95p --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft95perc_baseR_cos_lr0.003_bs256/checkpoint_best.pth; \
python eval_ensem.py --exp-id=imagenet_inv_ft95p --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft95perc_inv_cos_lr0.003_bs256/checkpoint_best.pth; \
python eval_ensem.py --exp-id=imagenet_eqR_ft95p --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft95perc_eqR_cos_lr0.003_bs256/checkpoint_best.pth; \
python eval_ensem.py --exp-id=imagenet_base_ft95p_onval --val_perc=5 --use_smaller_split_val --eval_on_train --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft95perc_baseR_cos_lr0.003_bs256/checkpoint_best.pth; \
python eval_ensem.py --exp-id=imagenet_inv_ft95p_onval --val_perc=5 --use_smaller_split_val --eval_on_train --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft95perc_inv_cos_lr0.003_bs256/checkpoint_best.pth; \
python eval_ensem.py --exp-id=imagenet_eqR_ft95p_onval --val_perc=5 --use_smaller_split_val --eval_on_train --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft95perc_eqR_cos_lr0.003_bs256/checkpoint_best.pth;

python eval_ensem.py --exp-id=imagenet_bei_ft95p_ontrain --val_perc=5 --use_smaller_split --eval_on_train --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft95perc_baseR_cos_lr0.003_bs256/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft95perc_inv_cos_lr0.003_bs256/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft95perc_eqR_cos_lr0.003_bs256/checkpoint_best.pth; \
python eval_ensem.py --exp-id=imagenet_base_ft95p_ontrain --val_perc=5 --use_smaller_split --eval_on_train --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft95perc_baseR_cos_lr0.003_bs256/checkpoint_best.pth; \
python eval_ensem.py --exp-id=imagenet_inv_ft95p_ontrain --val_perc=5 --use_smaller_split --eval_on_train --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft95perc_inv_cos_lr0.003_bs256/checkpoint_best.pth; \
python eval_ensem.py --exp-id=imagenet_eqR_ft95p_ontrain --val_perc=5 --use_smaller_split --eval_on_train --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft95perc_eqR_cos_lr0.003_bs256/checkpoint_best.pth;


python eval_ensem.py --exp-id=imagenet_bei_ft --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_baseR_cos_lr0.003_bs256/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_inv_cos_lr0.003_bs256/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_eqR_cos_lr0.003_bs256/checkpoint_best.pth;
python eval_ensem.py --exp-id=imagenet_base_ft --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_baseR_cos_lr0.003_bs256/checkpoint_best.pth; \
python eval_ensem.py --exp-id=imagenet_inv_ft --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_inv_cos_lr0.003_bs256/checkpoint_best.pth; \
python eval_ensem.py --exp-id=imagenet_eqR_ft --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_eqR_cos_lr0.003_bs256/checkpoint_best.pth;

python eval_ensem_calib.py --exp-id=ts_base_ft0.003_95_512 --batch-size=512 --val_perc=5 --server=aimos --gpu=0 --eval-mode=temp_scale --dataset=imagenet  --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet/ --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft95perc_baseR_cos_lr0.003_bs256/checkpoint_best.pth
python eval_ensem_gate2.py --exp-id=testgatesharp --sharpen_T=0.1 --gpu=0 --server=aimos --gate_arch=smallmlp --me_max --lmbd=0.0 --use_eps --lr-scheduler=cosine --gate=frozen --lr-gate=0.0001 --dataset=imagenet --combine_sep_ckpts --use_default_pretrained

python eval_ensem.py --exp-id=imagenet_eq42_ft --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_eq42_cos_lr0.003_bs256/checkpoint_best.pth; \
python eval_ensem.py --exp-id=imagenet_eq69_ft --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_eq69_cos_lr0.003_bs256/checkpoint_best.pth; \
python eval_ensem.py --exp-id=imagenet_eq31_ft --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_eq31_cos_lr0.003_bs256/checkpoint_best.pth; \
python eval_ensem.py --exp-id=imagenet_eq24_ft --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_eq24_cos_lr0.003_bs256/checkpoint_best.pth; \
python eval_ensem.py --exp-id=imagenet_eqR_fts1 --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_eqR_cos_lr0.003_bs256_ftseed1/checkpoint_best.pth; \
python eval_ensem.py --exp-id=imagenet_eqR_fts2 --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_eqR_cos_lr0.003_bs256_ftseed2/checkpoint_best.pth; \
python eval_ensem.py --exp-id=imagenet_eqR_fts3 --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_eqR_cos_lr0.003_bs256_ftseed3/checkpoint_best.pth;


python eval_ensem.py --exp-id=imagenet_bei_ft_2 --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_baseR_cos_lr0.003_bs256/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_inv_cos_lr0.003_bs256/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_eqR_cos_lr0.003_bs256/checkpoint_best.pth;
python eval_ensem.py --exp-id=imagenet_bei_ft95p_onval_2 --val_perc=5 --use_smaller_split_val --eval_on_train --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft95perc_baseR_cos_lr0.003_bs256/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft95perc_inv_cos_lr0.003_bs256/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft95perc_eqR_cos_lr0.003_bs256/checkpoint_best.pth; \
python eval_ensem.py --exp-id=imagenet_bei_ft95p_2 --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft95perc_baseR_cos_lr0.003_bs256/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft95perc_inv_cos_lr0.003_bs256/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft95perc_eqR_cos_lr0.003_bs256/checkpoint_best.pth;

python eval_ensem.py --exp-id=imagenet_eee_ftseeds --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_eqR_cos_lr0.003_bs256_ftseed1/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_eqR_cos_lr0.003_bs256_ftseed2/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_eqR_cos_lr0.003_bs256_ftseed3/checkpoint_best.pth
python eval_ensem.py --exp-id=imagenet_eee --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_eqR_cos_lr0.003_bs256/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_eq69_cos_lr0.003_bs256/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_eq42_cos_lr0.003_bs256/checkpoint_best.pth

python eval_ensem.py --exp-id=testfold4 --fold=4 --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_eqR_cos_lr0.003_bs256/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_eq69_cos_lr0.003_bs256/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_eq42_cos_lr0.003_bs256/checkpoint_best.pth
python eval_ensem.py --exp-id=imagenet_bie_ft_w0.34_0.28_0.38 --weighting 0.34 0.28 0.38 --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_baseR_cos_lr0.003_bs256/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_inv_cos_lr0.003_bs256/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_eqR_cos_lr0.003_bs256/checkpoint_best.pth;

python eval_ensem_gate.py --gate_arch=smallmlp --smallmlp_hd=128 --use_eps --exp-id=test9 --dataset=imagenet --server=sc --eval-mode=freeze --gpu=0 --dataset=imagenet --combine_sep_ckpts --use_default_pretrained
python eval_ensem_gate.py --gate_arch=smallmlp --smallmlp_hd=128 --use_eps --exp-id=test9 --dataset=imagenet --server=sc --eval-mode=freeze --gpu=0 --dataset=imagenet --combine_sep_ckpts --use_default_pretrained

python eval_ensem_gate.py --server=sc --submit --arg_str="--gate_arch=smallmlp --smallmlp_hd=128 --me_max --lmbd=${lmbd} --use_eps --lr-scheduler=cosine --gate=frozen --lr-gate=${lr} --eval_subset100 --dataset=imagenet --combine_sep_ckpts --use_default_pretrained "

python eval_ensem_db.py --exp-id=imagenet_bie_db --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_baseR_cos_lr0.003_bs256/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_inv_cos_lr0.003_bs256/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_eqR_cos_lr0.003_bs256/checkpoint_best.pth;
python eval_ensem.py --exp-id=imagenet_bie_ --val_perc=5 --use_smaller_split --eval_on_train  --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_baseR_cos_lr0.003_bs256/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_inv_cos_lr0.003_bs256/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_eqR_cos_lr0.003_bs256/checkpoint_best.pth;



python eval_ensem.py --exp-id=imagenet_bei_ft95p_ontrain2 --val_perc=5 --use_smaller_split --eval_on_train --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft95perc_baseR_cos_lr0.003_bs256/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft95perc_inv_cos_lr0.003_bs256/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft95perc_eqR_cos_lr0.003_bs256/checkpoint_best.pth

python eval_ensem_gate.py --exp-id=int_gate_adam_cew --optim=adam --lr-gate=0.001 --weight-decay=5e-4 --server=aimos --gpu=0 --validate_freq=20 --gate_arch=mlp_bn4w --gate=frozen --dataset=imagenet --val_perc=5 --use_smaller_split_val --gate_loss=cew_softmax --batch-size=256 --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft95perc_baseR_cos_lr0.003_bs256/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft95perc_inv_cos_lr0.003_bs256/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft95perc_eqR_cos_lr0.003_bs256/checkpoint_best.pth;

python eval_ensem.py --exp-id=imagenet_eee_ft --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_eqR_cos_lr0.003_bs256/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_eq69_cos_lr0.003_bs256/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_eq42_cos_lr0.003_bs256/checkpoint_best.pth; \
python eval_ensem.py --exp-id=imagenet_bie_ft --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_baseR_cos_lr0.003_bs256/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_inv_cos_lr0.003_bs256/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_eqR_cos_lr0.003_bs256/checkpoint_best.pth;


python eval_ensem.py --exp-id=imagenet_bie_ft95p_onval --val_perc=5 --use_smaller_split_val --eval_on_train --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft95perc_baseR_cos_lr0.003_bs256/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft95perc_inv_cos_lr0.003_bs256/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft95perc_eqR_cos_lr0.003_bs256/checkpoint_best.pth;


python eval_ensem.py --exp-id=testsplit --train_val_split=0 --eval-mode=finetune --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft95perc_baseR_cos_lr0.003_bs256/checkpoint_best.pth

python eval_ensem_calib.py --exp-id=ts_base_ft0.003_95_512 --batch-size=512 --val_perc=5 --server=aimos --gpu=0 --eval-mode=temp_scale --dataset=imagenet  --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet/ --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft95perc_baseR_cos_lr0.003_bs256/checkpoint_best.pth

for lr in 0.03 0.01 0.1 0.003
do
python eval_ensem.py --submit --server=sc --arg_str="--combine_sep_ckpt --lr-backbone=${lr} --lr-classifier=${lr} --arch=cifar_resnet18 --eval-mode=finetune --dataset=ftcifar100 --pretrained /home/gridsan/cloh/MAML-Soljacic/charlotte/cifar100_pt_ckpts/2023pt_datasetcifar100lr0.03fp16seed42/800.pth "
done

python eval_ensem.py --exp-id=imagenet_bbb_ftseeds --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_baseR_cos_lr0.003_bs256_ftseed1/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_baseR_cos_lr0.003_bs256_ftseed2/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_baseR_cos_lr0.003_bs256_ftseed3/checkpoint_best.pth

python eval_ensem.py --exp-id=imagenet_bbb --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_base69_cos_lr0.003_bs256/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_base31_cos_lr0.003_bs256/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_base24_cos_lr0.003_bs256/checkpoint_best.pth

python eval_ensem_calib.py --exp-id=ts_base_ft0.01_split1 --train_val_split=1 --server=aimos --gpu=0 --eval-mode=temp_scale --dataset=imagenet  --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet/ --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_base_e100_split1_cos_lr0.01_bs256/checkpoint_best.pth
# TO RUN
python eval_ensem_calib.py --exp-id=ts_eq_ft0.01_split1 --train_val_split=1 --server=aimos --gpu=0 --eval-mode=temp_scale --dataset=imagenet  --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet/ --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_eq_e100_split1_cos_lr0.01_bs256/checkpoint_best.pth
python eval_ensem_calib.py --exp-id=ts_inv_ft0.01_split1 --train_val_split=1 --server=aimos --gpu=0 --eval-mode=temp_scale --dataset=imagenet  --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet/ --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_inv_e100_split1_cos_lr0.01_bs256/checkpoint_best.pth



# python eval_ensem_calib.py --server=aimos --submit --arg_str="--train_val_split=1 --server=aimos --gpu=0 --eval-mode=temp_scale --dataset=imagenet  --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet/ --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_base_e100_split1_cos_lr0.01_bs256/checkpoint_best.pth
python eval_ensem.py --exp-id=test_cifar100 --server=aimos --gpu=0 --dataset=cifar100 --eval-mode=finetune --combine_sep_ckpts --data /gpfs/u/home/BNSS/BNSSlhch/scratch/datasets/ --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_pt_checkpoints/simclr-base-seedR-IN1k-800e-resnet50.pth

python eval_ensem.py --exp-id=test_flowers2 --batch-size=64 --server=aimos --gpu=0 --dataset=flowers-102 --eval-mode=finetune --combine_sep_ckpts --data /gpfs/u/home/BNSS/BNSSlhch/scratch/datasets/ --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_pt_checkpoints/simclr-base-seedR-IN1k-800e-resnet50.pth

python eval_ensem.py --server=sc --gpu=0 --exp-id=testcifarft100 --lr-backbone=0.01 --lr-classifier=0.01 --arch=cifar_resnet18 --eval-mode=finetune --dataset=ftcifar100 --combine_sep_ckpt --pretrained /home/gridsan/cloh/MAML-Soljacic/charlotte/cifar100_pt_ckpts/2023pt_datasetcifar100lr0.03fp16seed42/800.pth

python eval_ensem.py --exp-id=test_smaller_split --use_smaller_split --val_perc=500 --eval-mode=freeze --gpu=0 --server=aimos --dataset=imagenet --combine_sep_ckpts --data /gpfs/u/locker/200/CADS/datasets/ImageNet  --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_base69_cos_lr0.003_bs256/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_base31_cos_lr0.003_bs256/checkpoint_best.pth /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/dist_models/ft_base24_cos_lr0.003_bs256/checkpoint_best.pth

python main_cifar.py --log_id=testft --server=sc --checkpoint_path="/home/gridsan/cloh/MAML-Soljacic/charlotte/cifar100_pt_ckpts/2023pt_datasetcifar100lr0.03fp16seed42/800.pth" --ft_dataset=cifar100 --ft_mode=ft --ft_lr=0.01

python eval_ensem.py --exp-id=testlpin200 --server=aimos --gpu=0 --dataset=imagenet --eval-mode=linear_probe --combine_sep_ckpts --lr-classifier=0.3 --lr-scheduler=cosine --use_smaller_split --val_perc=200 --data /gpfs/u/locker/200/CADS/datasets/ImageNet/ --pretrained /gpfs/u/home/BNSS/BNSSlhch/scratch/ensem_ssl/saved_models/simclr-inv-seed42-IN200_400e_bs4096/checkpoint.pth
