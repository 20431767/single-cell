#$repeat_num=10
$repeat_num=1
#$data_dir="J:\cgi_datasets"
$data_dir="H:\data\data\"
$dataset_dirs="cellbench_kolod_pollen"
#$dataset_dirs="human_pancreas", "human_progenitor"
#$dataset_dirs="human_pancreas", "human_progenitor", "mouse_cortex", "human_melanoma", "mouse_stem", "mouse_pancreas", "mouse_blastomeres", "human_cell_line", "human_colorectal_cancer", "mouse_embryos", "mouse_pancreatic_circulating_tumor", "human_embryos"
#$dataset_dirs='mars_skin', 'mars_limb_muscle', 'mars_spleen', 'mars_trachea', 'mars_tongue', 'mars_thymus', 'mars_bladder', 'mars_liver', 'mars_mammary_gland'
#$dataset_dirs='mars_limb_muscle', 'mars_spleen', 'mars_trachea', 'mars_thymus', 'mars_mammary_gland'
#$dataset_dirs="mouse_cortex", "mouse_stem", "human_progenitor", "human_melanoma", "human_pancreas", "mouse_pancreas", "mouse_blastomeres", "human_cell_line", "human_colorectal_cancer", "mouse_embryos", "mouse_pancreatic_circulating_tumor", "human_embryos"
#$dataset_dirs="mouse_cortex", "mouse_stem", "human_melanoma", "human_pancreas", "mouse_pancreas", "mouse_blastomeres", "human_cell_line", "human_colorectal_cancer", "mouse_embryos", "mouse_pancreatic_circulating_tumor", "human_embryos", 'mars_skin', 'mars_limb_muscle', 'mars_trachea', 'mars_tongue', 'mars_thymus', 'mars_bladder', 'mars_liver', 'mars_mammary_gland', "mars_spleen"

#$dataset_dirs="mouse_cortex"
<# $lambda_bs="0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"
$lambda_cs="0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"
 #>
# "0.1", "0.3", "0.5", "0.7", "0.9"
#$lambda_bs="0", "0.2", "0.4", "0.6", "0.8", "1.0"
#$lambda_cs="0", "0.2", "0.4", "0.6", "0.8", "1.0"
#$lambda_bs="0.4", "0.6", "0.8", "1.0"
#$lambda_cs="0", "0.05", "0.1", "0.15", "0.2", "0.25", "0.3"
$lambda_bs="0"
$lambda_cs="0"
$lambda_ds="0"
$lambda_es="0.1"
#$lambda_ds="0"
#$epochs=50
$epochs=10
$hg=0.1
#$dims=25, 100, 400, 1000
$dims=400
#$dims=2..30
#$dims="0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"
#cd "F:/OneDrive - Hong Kong Baptist University/year1_1/ge_code2"
#$linkage="complete", "ward", "average", "single"

$methods="KMeans" # , "AgglomerativeClustering" #"SpectralClustering"
foreach ($lambda_b in $lambda_bs) {
    foreach ($lambda_c in $lambda_cs) {
        foreach($lambda_d in $lambda_ds) {
            foreach($lambda_e in $lambda_es) {
                $output_dir="test1-entropy-mask-lambda_b_$lambda_b-lambda_c_$lambda_c-lambda_d_$lambda_d"
                foreach ($dim in $dims) {
                    conda activate cgi2
                    foreach ($method in $methods) {
                        foreach ($dataset_dir in $dataset_dirs) {
                            if (!(Test-Path -Path "$data_dir/$dataset_dir/$output_dir")) {
                                New-Item -Path "$data_dir/$dataset_dir" -Name "$output_dir" -ItemType "directory"
                            }
                            $exp_name="hg$hg-dim$dim-reinit-$method"
                            if(Test-Path "$data_dir/$dataset_dir/$output_dir/$exp_name.cluster.txt")
                            {
                                Remove-Item "$data_dir/$dataset_dir/$output_dir/$exp_name.cluster.txt"
                            }
                            if(Test-Path "$data_dir/$dataset_dir/$output_dir/$exp_name.anno.txt")
                            {
                                Remove-Item "$data_dir/$dataset_dir/$output_dir/$exp_name.anno.txt"
                            }
                            foreach($i in 1..$repeat_num) {
                                # only one should
                                # $exp_name="$dim-reinit-$method"
                                python run.py --dim $dim --inherit-centroids --data-dir $data_dir --bench-dataset $dataset_dir --output-dir $output_dir --output-ari-file "$exp_name.ari.txt" --seed $i --method $method --lambda_b $lambda_b --lambda_c $lambda_c --lambda_d $lambda_d --lambda_e $lambda_e --epochs $epochs
                                # $exp_name="$dim-inherit-$method"
                                # python run.py --dim $dim --inherit-centroids --data-dir $data_dir --bench-dataset $dataset_dir --output-ari-file "$exp_name.ari.txt" --seed $i --method $method
                                # $exp_name="$dim-reinit-$method-L2"
                                # python run.py --dim $dim --data-dir $data_dir --bench-dataset $dataset_dir --output-ari-file "$exp_name.ari.txt" --seed $i --method $method --L2-normalization
                                # $exp_name="$dim-inherit-$method-L2"
                                # python run.py --dim $dim --inherit-centroids --data-dir $data_dir --bench-dataset $dataset_dir --output-ari-file "$exp_name.ari.txt" --seed $i --method $method --L2-normalization
                            }
                        }
                    }
                    foreach ($method in $methods) {
                        foreach ($dataset_dir in $dataset_dirs) {
                            if (!(Test-Path -Path "$data_dir/$dataset_dir/$output_dir")) {
                                New-Item -Path "$data_dir/$dataset_dir" -Name "$output_dir" -ItemType "directory"
                            }
                            $exp_name="hg$hg-dim$dim-inherit-$method"
                            if(Test-Path "$data_dir/$dataset_dir/$output_dir/$exp_name.cluster.txt")
                            {
                                Remove-Item "$data_dir/$dataset_dir/$output_dir/$exp_name.cluster.txt"
                            }
                            if(Test-Path "$data_dir/$dataset_dir/$output_dir/$exp_name.anno.txt")
                            {
                                Remove-Item "$data_dir/$dataset_dir/$output_dir/$exp_name.anno.txt"
                            }
                            foreach($i in 1..$repeat_num) {
                                # only one should
                                # $exp_name="$dim-reinit-$method"
                                # python run.py --dim $dim --data-dir $data_dir --bench-dataset $dataset_dir --output-ari-file "$exp_name.ari.txt" --seed $i --method $method
                                # $exp_name="$dim-inherit-$method"
                                #python run.py --dim $dim --inherit-centroids --data-dir $data_dir --bench-dataset $dataset_dir --output-dir $output_dir --output-ari-file "$exp_name.ari.txt" --seed $i --method $method --lambda_b $lambda_b --lambda_c $lambda_c --lambda_d $lambda_d --epochs $epochs --generate-files
                                # $exp_name="$dim-reinit-$method-L2"
                                # python run.py --dim $dim --data-dir $data_dir --bench-dataset $dataset_dir --output-ari-file "$exp_name.ari.txt" --seed $i --method $method --L2-normalization
                                # $exp_name="$dim-inherit-$method-L2"
                                # python run.py --dim $dim --inherit-centroids --data-dir $data_dir --bench-dataset $dataset_dir --output-ari-file "$exp_name.ari.txt" --seed $i --method $method --L2-normalization
                            }
                        }
                    }
                    if ($TRUE) {
                    # if ($dim -ge 200) {
                        foreach ($method in $methods) {
                            foreach ($dataset_dir in $dataset_dirs) {
                                if (!(Test-Path -Path "$data_dir/$dataset_dir/$output_dir")) {
                                    New-Item -Path "$data_dir/$dataset_dir" -Name "$output_dir" -ItemType "directory"
                                }
                                $exp_name="hg$hg-dim$dim-reinit-$method-L2"
                                if(Test-Path "$data_dir/$dataset_dir/$output_dir/$exp_name.cluster.txt")
                                {
                                    Remove-Item "$data_dir/$dataset_dir/$output_dir/$exp_name.cluster.txt"
                                }
                                if(Test-Path "$data_dir/$dataset_dir/$output_dir/$exp_name.anno.txt")
                                {
                                    Remove-Item "$data_dir/$dataset_dir/$output_dir/$exp_name.anno.txt"
                                }
                                foreach($i in 1..$repeat_num) {
                                    # only one should
                                    # $exp_name="$dim-reinit-$method"
                                    # python run.py --dim $dim --data-dir $data_dir --bench-dataset $dataset_dir --output-ari-file "$exp_name.ari.txt" --seed $i --method $method
                                    # $exp_name="$dim-inherit-$method"
                                    # python run.py --dim $dim --inherit-centroids --data-dir $data_dir --bench-dataset $dataset_dir --output-ari-file "$exp_name.ari.txt" --seed $i --method $method
                                    # $exp_name="$dim-reinit-$method-L2"
                                    #python run.py --dim $dim --data-dir $data_dir --bench-dataset $dataset_dir --output-dir $output_dir --output-ari-file "$exp_name.ari.txt" --seed $i --method $method --L2-normalization --lambda_b $lambda_b --lambda_c $lambda_c --lambda_d $lambda_d --epochs $epochs --generate-files
                                    # $exp_name="$dim-inherit-$method-L2"
                                    # python run.py --dim $dim --inherit-centroids --data-dir $data_dir --bench-dataset $dataset_dir --output-ari-file "$exp_name.ari.txt" --seed $i --method $method --L2-normalization
                                }
                            }
                        }
                        
                        foreach ($method in $methods) {
                            foreach ($dataset_dir in $dataset_dirs) {
                                if (!(Test-Path -Path "$data_dir/$dataset_dir/$output_dir")) {
                                    New-Item -Path "$data_dir/$dataset_dir" -Name "$output_dir" -ItemType "directory"
                                }
                                $exp_name="hg$hg-dim$dim-inherit-$method-L2"
                                if(Test-Path "$data_dir/$dataset_dir/$output_dir/$exp_name.cluster.txt")
                                {
                                    Remove-Item "$data_dir/$dataset_dir/$output_dir/$exp_name.cluster.txt"
                                }
                                if(Test-Path "$data_dir/$dataset_dir/$output_dir/$exp_name.anno.txt")
                                {
                                    Remove-Item "$data_dir/$dataset_dir/$output_dir/$exp_name.anno.txt"
                                }
                                foreach($i in 1..$repeat_num) {
                                    # only one should
                                    # $exp_name="$dim-reinit-$method"
                                    # python run.py --dim $dim --data-dir $data_dir --bench-dataset $dataset_dir --output-ari-file "$exp_name.ari.txt" --seed $i --method $method
                                    # $exp_name="$dim-inherit-$method"
                                    # python run.py --dim $dim --inherit-centroids --data-dir $data_dir --bench-dataset $dataset_dir --output-ari-file "$exp_name.ari.txt" --seed $i --method $method
                                    # $exp_name="$dim-reinit-$method-L2"
                                    # python run.py --dim $dim --data-dir $data_dir --bench-dataset $dataset_dir --output-ari-file "$exp_name.ari.txt" --seed $i --method $method --L2-normalization
                                    # $exp_name="$dim-inherit-$method-L2"
                                    #python run.py --dim $dim --inherit-centroids --data-dir $data_dir --bench-dataset $dataset_dir --output-dir $output_dir --output-ari-file "$exp_name.ari.txt" --seed $i --method $method --L2-normalization --lambda_b $lambda_b --lambda_c $lambda_c --lambda_d $lambda_d --epochs $epochs --generate-files
                                }
                            }
                        }
                    }
                }
                # $exp_name="$dim-inherit-centroids"
                $exp_name="hg$hg"
                # conda activate desc
            <#     foreach ($dataset_dir in $dataset_dirs) {
                    if (!(Test-Path -Path "$data_dir/$dataset_dir/$output_dir")) {
                        New-Item -Path "$data_dir/$dataset_dir" -Name "$output_dir" -ItemType "directory"
                    }
                    Get-ChildItem "result_tmp" -Recurse | Remove-Item
                    Remove-Item "$data_dir/$dataset_dir/$output_dir/$exp_name.desc.ari.txt"
                    foreach($i in 1..$repeat_num) {
                        python Py_benchmark_desc.py --data-dir $data_dir --bench-dataset $dataset_dir --output-dir $output_dir --output-ari-file "$exp_name.desc.ari.txt" --seed $i --hg $hg
                    }
                } #>


            <#     cd "F:/OneDrive - Hong Kong Baptist University/year1_1/compare/DRjCC"
                foreach ($dataset_dir in $dataset_dirs) {
                    if (!(Test-Path -Path "$data_dir/$dataset_dir/$output_dir")) {
                        New-Item -Path "$data_dir/$dataset_dir" -Name "$output_dir" -ItemType "directory"
                    }
                    Remove-Item "$data_dir/$dataset_dir/$output_dir/$exp_name.drjcc.cluster.txt"
                    Remove-Item "$data_dir/$dataset_dir/$output_dir/$exp_name.drjcc.anno.txt"
                    Remove-Item "./run_local_fun.m"
                    Add-Content -Path "./run_local_fun.m" -Value "MATLAB_benchmark_DRjCC('$data_dir', '$dataset_dir', '$exp_name', 100, '$hg', $repeat_num)"
                    # no need to repeat
                    matlab -nosplash -nodesktop -nojvm -batch run_local_fun
                } #>


            <#     cd ~/cgi
                foreach ($dataset_dir in $dataset_dirs) {
                    if (!(Test-Path -Path "$data_dir/$dataset_dir/$output_dir")) {
                        New-Item -Path "$data_dir/$dataset_dir" -Name "$output_dir" -ItemType "directory"
                    }
                    Remove-Item "$data_dir/$dataset_dir/$output_dir/$exp_name.raceid3.cluster.txt"
                    Remove-Item "$data_dir/$dataset_dir/$output_dir/$exp_name.sc3.cluster.txt"
                    Remove-Item "$data_dir/$dataset_dir/$output_dir/$exp_name.monocle2.cluster.txt"
                    Remove-Item "$data_dir/$dataset_dir/$output_dir/$exp_name.seurat.cluster.txt"
                    Remove-Item "$data_dir/$dataset_dir/$output_dir/$exp_name.simlr.cluster.txt"
                    Remove-Item "$data_dir/$dataset_dir/$output_dir/$exp_name.raceid3.anno.txt"
                    Remove-Item "$data_dir/$dataset_dir/$output_dir/$exp_name.sc3.anno.txt"
                    Remove-Item "$data_dir/$dataset_dir/$output_dir/$exp_name.monocle2.anno.txt"
                    Remove-Item "$data_dir/$dataset_dir/$output_dir/$exp_name.seurat.anno.txt"
                    Remove-Item "$data_dir/$dataset_dir/$output_dir/$exp_name.simlr.anno.txt"
                    foreach($i in 1..$repeat_num) {
                        Rscript C:/Users/cskexu/Documents/R_benchmark.R $data_dir $dataset_dir $exp_name $i $hg
                    }
                } #>


            <#     $exp_name="$exp_name-defaults"
                foreach ($dataset_dir in $dataset_dirs) {
                    if (!(Test-Path -Path "$data_dir/$dataset_dir/$output_dir")) {
                        New-Item -Path "$data_dir/$dataset_dir" -Name "$output_dir" -ItemType "directory"
                    }
                    Remove-Item "$data_dir/$dataset_dir/$output_dir/$exp_name.raceid3.ari.txt"
                    Remove-Item "$data_dir/$dataset_dir/$output_dir/$exp_name.sc3.ari.txt"
                    Remove-Item "$data_dir/$dataset_dir/$output_dir/$exp_name.seurat.ari.txt"
                    Remove-Item "$data_dir/$dataset_dir/$output_dir/$exp_name.simlr.ari.txt"
                    foreach($i in 1..$repeat_num) {
                        Rscript C:/Users/cskexu/Documents/R_benchmark_defaults.R $data_dir $dataset_dir $exp_name $i $hg
                    }
                } #>
            }
        }
    }
}
#cd "F:/OneDrive - Hong Kong Baptist University/year1_1/ge_code2"
