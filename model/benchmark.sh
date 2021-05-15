repeat_num=10
epochs=30
hg=0.1
repeat_num=1
#data_dir="/home/comp/cskexu/cgi_datasets"
data_dir="/d/HKBU/Course/Research/Project/cgi_datasets"
declare -a dataset_dirs=("human_pancreas" "human_progenitor" "mouse_cortex" "human_melanoma" "mouse_stem" "mouse_pancreas" "mouse_blastomeres" "human_cell_line" "human_colorectal_cancer" "mouse_embryos" "mouse_pancreatic_circulating_tumor" "human_embryos")
declare -a dataset_dirs=("human_colorectal_cancer")  # "human_pancreas" "mouse_pancreas"
declare -a repeats=($(seq 1 1 $repeat_num))
#declare -a dims=($(seq 0.1 0.1 1.0))
declare -a dims=("400")
declare -a lambda_bs=("0")
declare -a lambda_cs=("0")
declare -a lambda_ds=("0" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0")
#cd ~/ge_code2
#conda activate cgi2
declare -a methods=("KMeans")
#declare -a methods=("SpectralClustering")
for lambda_b in "${lambda_bs[@]}"
do
    for lambda_c in "${lambda_cs[@]}"
        do
        for lambda_d in "${lambda_ds[@]}"
        do
            for dim in "${dims[@]}"
            do
                for method in "${methods[@]}"
                do
                    exp_name="hg$hg-dim$dim-reinit-$method"
                    for dataset_dir in "${dataset_dirs[@]}"
                    do
                        if [ ! -d "$data_dir/$dataset_dir/output" ]
                        then
                            mkdir "$data_dir/$dataset_dir/output"
                        fi
                        
						if [ -f "$data_dir/$dataset_dir/output/$exp_name.cluster.txt" ]
                        then
							rm "$data_dir/$dataset_dir/output/$exp_name.cluster.txt"
						fi
						
						if [ -f "$data_dir/$dataset_dir/output/$exp_name.anno.txt" ]
                        then
							rm "$data_dir/$dataset_dir/output/$exp_name.anno.txt"
						fi
                        for i in "${repeats[@]}"
                        do
                            python run.py --dim $dim --data-dir $data_dir --bench-dataset $dataset_dir --output-dir $output_dir --output-ari-file "$exp_name.ari.txt" --seed $i --method $method --lambda_b $lambda_b --lambda_c $lambda_c --epochs $epochs
                             # python run.py --dim $dim --data-dir $data_dir --bench-dataset $dataset_dir --output-ari-file "$exp_name.ari.txt" --seed $i --method $method --lambda_b $lambda_b --lambda_c $lambda_c --lambda_d $lambda_d --epochs $epochs
                        done
                    done
                done
				
				for method in "${methods[@]}"
                do
                    exp_name="hg$hg-dim$dim-inherit-$method"
                    for dataset_dir in "${dataset_dirs[@]}"
                    do
                        if [ ! -d "$data_dir/$dataset_dir/output" ]
                        then
                            mkdir "$data_dir/$dataset_dir/output"
                        fi
						if [ -f "$data_dir/$dataset_dir/output/$exp_name.cluster.txt" ]
                        then
							rm "$data_dir/$dataset_dir/output/$exp_name.cluster.txt"
						fi
						
						if [ -f "$data_dir/$dataset_dir/output/$exp_name.anno.txt" ]
                        then
							rm "$data_dir/$dataset_dir/output/$exp_name.anno.txt"
						fi
						
                        for i in "${repeats[@]}"
                        do
							python run.py --dim $dim --inherit-centroids --data-dir $data_dir --bench-dataset $dataset_dir --output-dir $output_dir --output-ari-file "$exp_name.ari.txt" --seed $i --method $method --lambda_b $lambda_b --lambda_c $lambda_c --epochs $epochs
                             # python run.py --dim $dim --data-dir $data_dir --bench-dataset $dataset_dir --output-ari-file "$exp_name.ari.txt" --seed $i --method $method --lambda_b $lambda_b --lambda_c $lambda_c --lambda_d $lambda_d --epochs $epochs
                        done
                    done
                done
				
				for method in "${methods[@]}"
                do
                    exp_name="hg$hg-dim$dim-reinit-$method-L2"
                    for dataset_dir in "${dataset_dirs[@]}"
                    do
                        if [ ! -d "$data_dir/$dataset_dir/output" ]
                        then
                            mkdir "$data_dir/$dataset_dir/output"
                        fi
						if [ -f "$data_dir/$dataset_dir/output/$exp_name.cluster.txt" ]
                        then
							rm "$data_dir/$dataset_dir/output/$exp_name.cluster.txt"
						fi
						
						if [ -f "$data_dir/$dataset_dir/output/$exp_name.anno.txt" ]
                        then
							rm "$data_dir/$dataset_dir/output/$exp_name.anno.txt"
						fi
						
                        for i in "${repeats[@]}"
                        do
							python run.py --dim $dim --data-dir $data_dir --bench-dataset $dataset_dir --output-dir $output_dir --output-ari-file "$exp_name.ari.txt" --seed $i --method $method --L2-normalization --lambda_b $lambda_b --lambda_c $lambda_c --epochs $epochs
                        done
                    done
                done
				
				for method in "${methods[@]}"
                do
                    exp_name="hg$hg-dim$dim-inherit-$method-L2"
                    for dataset_dir in "${dataset_dirs[@]}"
                    do
                        if [ ! -d "$data_dir/$dataset_dir/output" ]
                        then
                            mkdir "$data_dir/$dataset_dir/output"
                        fi
						if [ -f "$data_dir/$dataset_dir/output/$exp_name.cluster.txt" ]
                        then
							rm "$data_dir/$dataset_dir/output/$exp_name.cluster.txt"
						fi
						
						if [ -f "$data_dir/$dataset_dir/output/$exp_name.anno.txt" ]
                        then
							rm "$data_dir/$dataset_dir/output/$exp_name.anno.txt"
						fi
						
                        for i in "${repeats[@]}"
                        do
							python run.py --dim $dim --inherit-centroids --data-dir $data_dir --bench-dataset $dataset_dir --output-dir $output_dir --output-ari-file "$exp_name.ari.txt" --seed $i --method $method --L2-normalization --lambda_b $lambda_b --lambda_c $lambda_c  --epochs $epochs
                        done
                    done
                done
            done
        done
    done
done


    # cd ~/cgi
    # for dataset_dir in $dataset_dirs
    # do
        # if [ ! -d "$data_dir/$dataset_dir/output" ]
        # then
            # mkdir "$data_dir/$dataset_dir/output"
        # fi
        # rm "$data_dir/$dataset_dir/output/$exp_name.raceid3.ari.txt"
        # rm "$data_dir/$dataset_dir/output/$exp_name.sc3.ari.txt"
        # rm "$data_dir/$dataset_dir/output/$exp_name.seurat.ari.txt"
        # rm "$data_dir/$dataset_dir/output/$exp_name.simlr.ari.txt"
        # for i in {1..10}
        # do
            # Rscript --save $exp_name
        # done
    # done
