for seed in 42 69 31 24 96 33
do
python main_cifar.py --submit --add_prefix=2023 --arg_str="--pt_dataset=cifar100 --tfm=rotate --tfm_mode=eq --lr=0.06 --lmbd=0.4 --fp16 --seed=${seed} "
python main_cifar.py --submit --add_prefix=2023 --arg_str="--pt_dataset=cifar100 --tfm=rotate --tfm_mode=inv --lr=0.03 --fp16 --seed=${seed} "
python main_cifar.py --submit --add_prefix=2023 --arg_str="--pt_dataset=cifar100 --lr=0.03 --fp16 --seed=${seed} "
done

for tfm in invert blur vflip
do
for seed in 42 69 31 24 96 33
do
python main_cifar.py --submit --add_prefix=2023 --arg_str="--pt_dataset=cifar100 --tfm=${tfm} --tfm_mode=eq --lr=0.06 --lmbd=0.4 --fp16 --seed=${seed} "
python main_cifar.py --submit --add_prefix=2023 --arg_str="--pt_dataset=cifar100 --tfm=${tfm} --tfm_mode=inv --lr=0.03 --fp16 --seed=${seed} "
done
done

for seed in 13 23 54 45
do
python main_cifar.py --submit --add_prefix=2023 --arg_str="--pt_dataset=cifar100 --tfm=rotate --tfm_mode=eq --lr=0.06 --lmbd=0.4 --fp16 --seed=${seed} "
python main_cifar.py --submit --add_prefix=2023 --arg_str="--pt_dataset=cifar100 --tfm=rotate --tfm_mode=inv --lr=0.03 --fp16 --seed=${seed} "
python main_cifar.py --submit --add_prefix=2023 --arg_str="--pt_dataset=cifar100 --lr=0.03 --fp16 --seed=${seed} "
done

python eval_cifar.py --ft_dataset=cifar100 --ft_mode=ft --ft_lr=0.003 --rrc=0.5 --fp16 --pretrained_id=pt_datasetcifar100fp16lr0.03seed43
python eval_cifar.py --ft_dataset=cifar100 1101ft_datasetcifar100ft_modeftft_lr0.003rrc0.5fp16pretrained_id.out

for lr in 0.01 0.02 0.04 0.06
do
for seed in 31
do
python main_cifar.py --submit --add_prefix=2023 --arg_str="--pt_dataset=cifar100 --tfm=rotate --tfm_mode=inv --lr=${lr} --fp16 --seed=${seed} "
done
done

for lr in 0.01 0.02 0.04 0.06
do
for seed in 31
do
python main_cifar.py --submit --add_prefix=2023 --arg_str="--pt_dataset=cifar100 --lr=${lr} --fp16 --seed=${seed} "
done
done


for lr in 0.02
do
for seed in 42 69 24 96 33
do
python main_cifar.py --submit --add_prefix=2023 --arg_str="--pt_dataset=cifar100 --tfm=rotate --tfm_mode=inv --lr=${lr} --fp16 --seed=${seed} "
done
done


for tfm in invert
do
for ptlr in 0.02 0.01 0.04 0.06 0.03 0.08
do
python main_cifar.py --submit --add_prefix=2023 --arg_str="--pt_dataset=cifar100 --tfm=${tfm} --tfm_mode=eq --lr=${ptlr} --lmbd=0.4 --fp16 --seed=54 "
python main_cifar.py --submit --add_prefix=2023 --arg_str="--pt_dataset=cifar100 --tfm=${tfm} --tfm_mode=inv --lr=${ptlr} --fp16 --seed=54 "
done
done

for tfm in jigsaw
do
for ptlr in 0.02 0.01 0.04 0.06 0.08 0.03
do
python main_cifar.py --submit --add_prefix=2023 --arg_str="--pt_dataset=cifar100 --tfm=${tfm} --tfm_mode=eq --lr=${ptlr} --lmbd=0.4 --fp16 --seed=54 "
python main_cifar.py --submit --add_prefix=2023 --arg_str="--pt_dataset=cifar100 --tfm=${tfm} --tfm_mode=inv --lr=${ptlr} --fp16 --seed=54 "
done
done


for seed in 54 45 13 23 32 67 76
do
python main_cifar.py --submit --add_prefix=2023 --arg_str="--pt_dataset=cifar100 --tfm=rotate --tfm_mode=eq --lr=0.06 --lmbd=0.4 --fp16 --seed=${seed} "
python main_cifar.py --submit --add_prefix=2023 --arg_str="--pt_dataset=cifar100 --tfm=rotate --tfm_mode=inv --lr=0.02 --fp16 --seed=${seed} "
python main_cifar.py --submit --add_prefix=2023 --arg_str="--pt_dataset=cifar100 --lr=0.03 --fp16 --seed=${seed} "
done
