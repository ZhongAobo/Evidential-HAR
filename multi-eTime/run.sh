source activate zwt
cd /media/xxxy/Data1/ZXY/zxy/eTime/AdaTimev4

down_array=('max' 'max_cnn' 'cnn' 'avg' 'rand')
alpha_array=(0.1 0.5 1.0 2.0 3.0 4.0 5.0)
method_array=('DDC')
dist_array=("dir_edl")
data_array=("WISDM")
lr_array=(5e-4 1e-3 2e-3 3e-3 5e-3 6e-3 8e-3)
for down in "${down_array[@]}"; do
    for lr in "${lr_array[@]}"; do
        for data in "${data_array[@]}"; do
            for dist in "${dist_array[@]}"; do
                for method in "${method_array[@]}"; do
                    train_dir=/media/xxxy/Data1/ZXY/zxy/eTime/AdaTimev4/logs_${down}/${data}/${method}/${dist}/train
                    test_dir=/media/xxxy/Data1/ZXY/zxy/eTime/AdaTimev4/logs_${down}/${data}/${method}/${dist}/new_test_50_05
                    mkdir -p ${train_dir}
                    mkdir -p ${test_dir}
                    for alpha in "${alpha_array[@]}"; do
                        export CUDA_VISIBLE_DEVICES=0 &&  python main.py  \
                        --phase train  \
                        --save_dir /media/xxxy/Data1/ZXY/zxy/eTime/results_v4/${data}/${method}/${dist}/${lr}_${alpha}x_${down} \
                        --data_path /media/xxxy/Data1/ZXY/zxy/eTime/dataset \
                        --da_method ${method} \
                        --dataset ${data} \
                        --backbone CNN \
                        --num_runs 1 \
                        --alpha ${alpha} \
                        --dist ${dist} \
                        --lr ${lr} --down ${down} \
                        2>&1 | tee  ${train_dir}/${lr}_${alpha}x.log

                        export CUDA_VISIBLE_DEVICES=0 &&  python main.py  \
                        --phase test  \
                        --save_dir /media/xxxy/Data1/ZXY/zxy/eTime/results_v4/${data}/${method}/${dist}/${lr}_${alpha}x_${down} \
                        --data_path /media/xxxy/Data1/ZXY/zxy/eTime/dataset \
                        --da_method ${method} \
                        --dataset ${data} \
                        --backbone CNN \
                        --num_runs 1 \
                        --alpha ${alpha} \
                        --dist ${dist} \
                        --lr ${lr} --down ${down} \
                        2>&1 | tee  ${test_dir}/${lr}_${alpha}x.log
                    done
                done
            done
        done
    done
done

