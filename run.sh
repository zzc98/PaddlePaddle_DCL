python main.py \
--gpus=1 \
--data=CUB \
--backbone=resnet50 \
--epoch=90 \
--T_max=60 \
--tb=8 \
--vb=8 \
--tnw=8 \
--vnw=8 \
--lr=0.001 \
--start_epoch=0 \
--detail=dcl_cub \
--size=512 \
--crop=448 \
--swap_num=7
#
#python main.py \
#--gpus=0 \
#--data=STCAR \
#--backbone=resnet50 \
#--epoch=90 \
#--T_max=60 \
#--tb=8 \
#--vb=8 \
#--tnw=8 \
#--vnw=8 \
#--lr=0.001 \
#--start_epoch=0 \
#--detail=dcl_cub \
#--size=512 \
#--crop=448 \
#--swap_num=7
#
#python main.py \
#--gpus=0 \
#--data=CUB \
#--backbone=resnet50 \
#--epoch=120 \
#--T_max=90 \
#--tb=8 \
#--vb=8 \
#--tnw=8 \
#--vnw=8 \
#--lr=0.001 \
#--start_epoch=0 \
#--detail=dcl_cub \
#--size=512 \
#--crop=448 \
#--swap_num=2
