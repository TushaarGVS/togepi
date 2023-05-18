from datasets import load_dataset

mnli = load_dataset("multi_nli")

#dataset.save_to_disk("/Users/lucasmolter/Documents/Cornell/Courses/Spring2023/CS5223 /togepi/Lucas/datatest")

#mnli.save_to_disk("/Users/lucasmolter/Documents/Cornell/Courses/Spring2023/CS5223 /togepi/Lucas/MNLI")

#dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

#dataset.save_to_disk("/mnt/beegfs/bulk/stripe/lm865/MNLI/")

mnli.save_to_disk("/mnt/beegfs/bulk/stripe/lm865/MNLI/")


