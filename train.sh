# 训练 VQGAN 第一阶段
torchrun --nproc_per_node=1 --master_port=4321 basicsr/train.py -opt options/VQGAN_512_ds32_nearest_stage1.yml --launcher pytorch
# # 训练完VQGAN后，可以通过下面代码预先获得训练数据集的密码本序列，从而加速后面阶段的训练过程:
# python scripts/generate_latent_gt.py
# # 训练密码本训练预测模块 第二阶段
# torchrun --nproc_per_node=1 --master_port=4322 basicsr/train.py -opt options/CodeFormer_stage2.yml --launcher pytorch
# # 训练可调模块 第三阶段
# torchrun --nproc_per_node=1 --master_port=4323 basicsr/train.py -opt options/CodeFormer_stage3.yml --launcher pytorch
