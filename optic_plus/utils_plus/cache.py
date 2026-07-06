import pickle, os, hashlib, json
from optic.utils import parameters

def cached_txtr(model_tx,
                param_tx,
                model_ch,
                param_ch,
                cache_dir="./cache"):
    os.makedirs(cache_dir, exist_ok=True)

    param = merge_params(param_tx, param_ch)
    key = hashlib.md5(json.dumps(vars(param), sort_keys=True, default=str).encode()).hexdigest()
    cache_file = f"{cache_dir}/txtr_{key}.pkl"

    if os.path.exists(cache_file):
        print("✅ Loading from cache...")
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    else:
        print("⚙️ Regenerate...")
        transmitter = model_tx(param_tx)
        sigWDM = model_ch(transmitter.sigTxWDM, param_ch)
        with open(cache_file, "wb") as f:
            pickle.dump(sigWDM, f)
        return transmitter, sigWDM

def merge_params(*params):
    merged = parameters()
    for p in params:
        vars(merged).update(vars(p))
    return merged