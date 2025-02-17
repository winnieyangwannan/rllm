### generate the code json

``` cd ~/rllm/rllm/data/preprocess ```

- run apps.ipynb to generate the apps.json

- run code_contests.ipynb generate the code_contests.json

- run codeforces.ipynb to generate codeforces.json
 ...
- run taco.ipynb to generate taco.josn

### generate the code parquet 

```cd ~/rllm/scripts/data ```
python3 code_dataset.py  --local_dir ~/rllm/rllm/data/preprocess