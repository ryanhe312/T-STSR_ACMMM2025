for dataset in LYSOSOME CCR5; do
    python codes/test.py  -v \
        -t Sakuya_arch_microscopy_mamba2 \
        -m experiments/LunaTokis_Mamba2_N2N_4k_${dataset}/models/latest_G.pth \
        -i datasets/${dataset}_test/lq/0001 \
        -r datasets/${dataset}_test/gt/ \
        -o results/${dataset}_mamba2
done