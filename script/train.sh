program="/content/drive/MyDrive/exe/preset_cv/code/src/main.py"

class_num=500000
vector_dim=512

batch_size=1000
epoch=0

separate_onehot=20
evaluate_batch_size=1000

# loss_type="nearest_orthogonal_loss"
# loss_type="nearest_orthogonal_or_more_loss"
loss_type="all_orthogonal_or_more_loss"
# loss_type="all_orthogonal_loss"
# loss_type="all_inverse_loss"

data_time=`date '+%Y%m%d%H%M%S'`

output_dir="/content/drive/MyDrive/exe/preset_cv/result/"${data_time}

mkdir ${output_dir}

log_path=${output_dir}"/log.log"

python3 ${program}\
        --class_num ${class_num}\
        --vector_dim ${vector_dim}\
        --batch_size ${batch_size}\
        --epoch ${epoch}\
        --loss_type ${loss_type}\
        --output_dir ${output_dir}\
        --separate_onehot ${separate_onehot}\
        --evaluate_batch_size ${evaluate_batch_size}\
        2>&1 | tee -a ${log_path}
