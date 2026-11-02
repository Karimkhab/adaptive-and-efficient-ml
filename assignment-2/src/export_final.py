"""Export the preselected map2 snapshot and its explicit inference time scale."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from ppo import load_checkpoint


def export(root):
    source=root/'ppo_v2_map2_safe_b/checkpoint_0600.npz'
    _,config=load_checkpoint(source)
    target=root/'ppo_v2_map2_final/checkpoint.npz'
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)
    config.update(inference_repeat=1,source_checkpoint=str(source),
        source_checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        selection='safe_b update600; dev512 and control frequency ablation; frozen before fresh test')
    target.with_suffix('.json').write_text(json.dumps(config,indent=2)+'\n')
    paths={'0':'ppo_v2_map0_seed0/checkpoint_1500.npz','1':'ppo_v2_map1_transfer_seed0/checkpoint_1000.npz',
           '2':'ppo_v2_map2_final/checkpoint.npz','3':'ppo_v2_map3_seed0/checkpoint_1000.npz'}
    (root/'final_models.json').write_text(json.dumps({'checkpoints':paths,
        'map2_control_frequency_hz':30,'map2_training_frequency_hz':5,
        'map2_fresh_test_seeds':[50000,50511]},indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('artifacts'))
    export(parser.parse_args().root)
