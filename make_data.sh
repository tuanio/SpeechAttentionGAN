for subset in train test valid; do
    echo $subset
    python processing_data.py \
        --src-domain clean --tgt-domain noisy \
        --path /home/stud_vantuan/share_with_150/cache/helicopter_1h_30m \
        --dest-path /home/stud_vantuan/data/data_pickle_gan/helicopter_1h_30m \
        --split $subset --stage $subset --threads 128 \
done