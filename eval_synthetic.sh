for dataset in RECEPTOR MICROTUBULE VESICLE; do
    for level in 3 4 7; do
        echo "Evaluating ${dataset} with noise level ${level}"
        # Evaluate the model
        python codes/test.py -v \
            -t Sakuya_arch_microscopy_mamba2 \
            -m experiments/LunaTokis_Mamba2_N2N_4k_${dataset}_snr_${level}/models/latest_G.pth \
            -i "datasets/${dataset}_snr_${level}_noisy_scale2/Out_HR_LR/LR/test_2/00001/*" \
            -o results/${dataset}_snr_${level}_noisy_scale2_n2n_mamba2 \
            -r datasets/${dataset}_snr_${level}_clean_scale2/Out_HR_LR/HR/test_2/00001
    done
done
