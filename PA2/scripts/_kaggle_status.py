from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
print(api.kernels_status("salmanrifa/pa2-sft-dpo-pipeline"))
