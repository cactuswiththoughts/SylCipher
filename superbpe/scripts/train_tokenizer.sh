dataset_name=olmo2_p99_truncate
do_whitespace_pretokenization=true
vocab_size=200000
num_bytes=$((10**10))
corpus_dir=/gscratch/xlab/alisaliu/pretokenization/data/${dataset_name}/train  # a directory containing txt files for tokenizer training

# convert num_bytes to something like 10G or 100M, depending on the value
if [ $num_bytes -ge $((10**9)) ]; then
    num_bytes_str=$(($num_bytes / 10**9))G
elif [ $num_bytes -ge $((10**6)) ]; then
    num_bytes_str=$(($num_bytes / 10**6))M
elif [ $num_bytes -ge $((10**3)) ]; then
    num_bytes_str=$(($num_bytes / 10**3))K
else
    num_bytes_str=${num_bytes}
fi

# convert vocab_size to something like 100K, depending on the value
if [ $vocab_size -ge 1000 ]; then
    vocab_size_str=$(($vocab_size / 1000))K
else
    vocab_size_str=${vocab_size}
fi

# if do_whitespace_pretokenization is true, set pretok_str to "pretok", else "nopretok"
if [ $do_whitespace_pretokenization == true ]; then
    pretok_str=pretok
else
    pretok_str=nopretok
fi

output_dir=tokenizer_json/${dataset_name}_${pretok_str}_${num_bytes_str}_${vocab_size_str}
echo "output_dir: $output_dir"

python -m train_tokenizer \
    --output_dir $output_dir \
    --corpus_dir $corpus_dir \
    --num_bytes $num_bytes \
    --vocab_size $vocab_size \
    --do_whitespace_pretokenization $do_whitespace_pretokenization
