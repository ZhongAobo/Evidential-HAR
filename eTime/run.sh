cd /media/xxxy/Data1/ZXY/zxy/Time/AdaTimev2

alpha_array=(0.1 0.5 1.0 2.0 3.0 5.0)
method_array=('Deep_Coral' 'MMDA' 'DANN' 'CDAN' 'DIRT' 'DSAN' 'HoMM' 'CoDATS' 'AdvSKM' 'SASA')
dist_array=("dir_edl")
data_array=("WISDM" "HAR" "HHAR")

for data in "${data_array[@]}"; do
    for dist in "${dist_array[@]}"; do
        for method in "${method_array[@]}"; do
            mkdir -p /media/xxxy/Data1/ZXY/zxy/Time/AdaTimev2/logs/${data}/${method}/${dist}/train
            mkdir -p /media/xxxy/Data1/ZXY/zxy/Time/AdaTimev2/logs/${data}/${method}/${dist}/test
            for alpha in "${alpha_array[@]}"; do
                export CUDA_VISIBLE_DEVICES=0 &&  python main.py  \
                --phase train  \
                --save_dir /media/fang/Data/ZXY/zxy/eTime/results/${data}/${method}/${dist}/${alpha}x \
                --data_path /media/fang/Data/ZXY/zxy/eTime/dataset \
                --da_method ${method} \
                --dataset ${data} \
                --backbone CNN \
                --num_runs 1 \
                --alpha ${alpha} \
                --dist ${dist} \
                2>&1 | tee  /media/xxxy/Data1/ZXY/zxy/Time/AdaTimev2/logs/${data}/${method}/${dist}/train/${alpha}x.log

                export CUDA_VISIBLE_DEVICES=0 &&  python main.py  \
                --phase test  \
                --save_dir /media/fang/Data/ZXY/zxy/eTime/results/${data}/${method}/${dist}/${alpha}x \
                --data_path /media/fang/Data/ZXY/zxy/eTime/dataset \
                --da_method ${method} \
                --dataset ${data} \
                --backbone CNN \
                --num_runs 1 \
                --alpha ${alpha} \
                --dist ${dist} \
                2>&1 | tee  /media/xxxy/Data1/ZXY/zxy/Time/AdaTimev2/logs/${data}/${method}/${dist}/test/${alpha}x.log
            done
        done
    done
done

