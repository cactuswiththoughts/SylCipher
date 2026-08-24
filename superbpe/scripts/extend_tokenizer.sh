dataset_name=olmo2_p99_truncate
orig_tokenizer_dir=tokenizer_json/${dataset_name}_pretok_10G_200K
num_inherit_merges=180000
vocab_size=200000

# create a str called num_inherit_merges_str, which turns 100000 into 100K
if [ $num_inherit_merges -ge 1000 ]; then
    num_inherit_merges_str=$(($num_inherit_merges / 1000))K
else
    num_inherit_merges_str=${num_inherit_merges}
fi

# convert vocab_size to something like 100K, depending on the value
if [ $vocab_size -ge 1000 ]; then
    vocab_size_str=$(($vocab_size / 1000))K
else
    vocab_size_str=${vocab_size}
fi

output_dir=tokenizer_json/${dataset_name}_10G_${num_inherit_merges_str}_extend_${vocab_size_str}_mw4_colon
echo "output_dir: $output_dir"

mkdir -p $output_dir
head -n $num_inherit_merges $orig_tokenizer_dir/merges.txt > $output_dir/merges.txt
cp $orig_tokenizer_dir/meta.json $output_dir/meta.json

python -m train_tokenizer \
    --output_dir $output_dir \
    --vocab_size $vocab_size \
    --do_whitespace_pretokenization false
